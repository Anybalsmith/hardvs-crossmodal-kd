import copy
import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.model import merge_dict

from mmaction.models import Recognizer2D
from mmaction.registry import MODELS
from mmaction.utils import SampleList


@MODELS.register_module()
class KDCrossModalRecognizer2D(Recognizer2D):
    """RGB -> Event cross-modal knowledge distillation for 2D recognizers.

    The recognizer itself is the student and receives the event frames
    (``inputs``). The teacher is a frozen recognizer built from the ``model``
    of ``teacher_config``, loaded from ``teacher_checkpoint``, and receives
    the RGB frames of the same recordings (``inputs_teacher``, provided by
    ``HARDVSPairedDataset``).

    Training loss of the student::

        cls_loss_weight        * CE(student, y)
      + kd_loss_weight         * mean_w[ KL(p_teacher^T || p_student^T) T^2 ]
      + feature_kd_loss_weight * mean_w[ feature_loss(f_student, f_teacher) ]

    where ``mean_w`` is a batch mean, weighted per sample by the teacher
    confidence (max softmax probability) when
    ``kd_sample_weight_mode='teacher_confidence'``.

    At validation / test time, when ``inputs_teacher`` is given, the
    prediction holds both scores (``pred_score = {'student', 'teacher'}``),
    so ``AccMetric`` reports ``acc/student_top1`` and ``acc/teacher_top1``.

    Args:
        teacher_config (str): Config file whose ``model`` builds the teacher.
        teacher_checkpoint (str): Checkpoint of the trained teacher.
        build_teacher (bool): If False, no teacher is built (student-only
            control run, requires ``kd_loss_weight=0`` and
            ``feature_kd_loss_weight=0``). Defaults to True.
        kd_temperature (float): Softmax temperature of the logit KD.
        kd_loss_weight (float): Weight of the logit KD loss.
        kd_loss_type (str): ``'kl'``, ``'hellinger'`` or
            ``'hellinger_squared'``.
        kd_sample_weight_mode (str): ``'none'`` or ``'teacher_confidence'``.
        teacher_confidence_temperature (float): Softmax temperature used to
            compute the teacher confidence weights.
        feature_kd_loss_weight (float): Weight of the feature KD loss.
        feature_kd_loss_type (str): ``'attention'`` (channel-agnostic
            attention transfer, works when student and teacher widths
            differ), ``'mse'``, ``'l1'`` or ``'cosine'``.
        apply_kd_weight_to_feature (bool): Also weight the feature KD loss
            per sample.
        cls_loss_weight (float): Weight of the student classification loss.
    """

    def __init__(self,
                 teacher_config: str,
                 teacher_checkpoint: Optional[str] = None,
                 build_teacher: bool = True,
                 kd_temperature: float = 4.0,
                 kd_loss_weight: float = 1.0,
                 kd_loss_type: str = 'kl',
                 kd_sample_weight_mode: str = 'none',
                 teacher_confidence_temperature: float = 1.0,
                 feature_kd_loss_weight: float = 0.0,
                 feature_kd_loss_type: str = 'attention',
                 apply_kd_weight_to_feature: bool = True,
                 cls_loss_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        if kd_sample_weight_mode not in (None, 'none', 'teacher_confidence'):
            raise ValueError(
                f'Unsupported kd_sample_weight_mode={kd_sample_weight_mode!r}'
                ". Expected 'none' or 'teacher_confidence'.")

        self.kd_temperature = kd_temperature
        self.kd_loss_weight = kd_loss_weight
        self.kd_loss_type = kd_loss_type
        self.kd_sample_weight_mode = kd_sample_weight_mode
        self.teacher_confidence_temperature = teacher_confidence_temperature
        self.feature_kd_loss_weight = feature_kd_loss_weight
        self.feature_kd_loss_type = feature_kd_loss_type
        self.apply_kd_weight_to_feature = apply_kd_weight_to_feature
        self.cls_loss_weight = cls_loss_weight
        self.teacher_checkpoint = teacher_checkpoint

        if not build_teacher:
            if kd_loss_weight > 0 or feature_kd_loss_weight > 0:
                raise ValueError(
                    'build_teacher=False is only valid for a student-only '
                    'control run: kd_loss_weight=0 and '
                    'feature_kd_loss_weight=0.')
            self.teacher = None
            return

        if teacher_checkpoint is None:
            raise ValueError('teacher_checkpoint is required to build the '
                             'teacher.')

        teacher_model = Config.fromfile(teacher_config).model

        # Building the teacher must not consume random numbers, so that the
        # student is initialised exactly as in the event-only baseline run
        # with the same seed.
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None)
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        try:
            self.teacher = MODELS.build(copy.deepcopy(teacher_model))
        finally:
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)

        # The teacher weights come from teacher_checkpoint: keep mmengine's
        # recursive init_weights from re-initialising it.
        self.teacher.init_weights = lambda *args, **kwargs: None
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self._teacher_loaded = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_weights(self):
        super().init_weights()
        if self.teacher is not None and not self._teacher_loaded:
            self._load_teacher_checkpoint(self.teacher_checkpoint)
            self._teacher_loaded = True

    def train(self, mode: bool = True):
        """Keep the teacher in eval mode."""
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _load_teacher_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Teacher checkpoint not found: {checkpoint_path!r}. Train the '
                'RGB baseline first and pass its checkpoint with --cfg-options '
                'model.teacher_checkpoint=<path>.')
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict = {
            (key[7:] if key.startswith('module.') else key): value
            for key, value in state_dict.items()
        }

        missing_keys, unexpected_keys = self.teacher.load_state_dict(
            state_dict, strict=False)
        if missing_keys:
            warnings.warn('Missing teacher keys when loading checkpoint: '
                          f'{missing_keys[:10]}')
        if unexpected_keys:
            warnings.warn('Unexpected teacher keys when loading checkpoint: '
                          f'{unexpected_keys[:10]}')

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def compute_kd_loss(self, student_logits: torch.Tensor,
                        teacher_logits: torch.Tensor) -> torch.Tensor:
        """Per-sample distillation loss between student and teacher logits."""
        temperature = self.kd_temperature
        student_probs = F.softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)

        if self.kd_loss_type == 'kl':
            return F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                teacher_probs,
                reduction='none').sum(dim=1) * (temperature * temperature)
        if self.kd_loss_type in ('hellinger', 'hellinger_squared'):
            eps = torch.finfo(student_probs.dtype).eps
            student_sqrt = torch.sqrt(student_probs.clamp_min(eps))
            teacher_sqrt = torch.sqrt(teacher_probs.clamp_min(eps))
            loss = 0.5 * (student_sqrt - teacher_sqrt).pow(2).sum(dim=1)
            if self.kd_loss_type == 'hellinger':
                loss = torch.sqrt(loss.clamp_min(eps))
            return loss
        raise ValueError(
            f'Unsupported kd_loss_type={self.kd_loss_type!r}. '
            "Expected 'kl', 'hellinger' or 'hellinger_squared'.")

    @staticmethod
    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        """Channel-agnostic spatial attention map (attention transfer)."""
        if feat.ndim == 2:
            attn = feat.abs()
        else:
            attn = feat.pow(2).mean(dim=1)
        attn = attn.reshape(attn.shape[0], -1)
        return F.normalize(attn, p=2, dim=1)

    def compute_feature_kd_loss(self, student_feats, teacher_feats,
                                batch_size: int) -> torch.Tensor:
        """Per-sample distillation loss between backbone features."""
        student_list = list(student_feats) if isinstance(
            student_feats, (list, tuple)) else [student_feats]
        teacher_list = list(teacher_feats) if isinstance(
            teacher_feats, (list, tuple)) else [teacher_feats]
        if len(student_list) != len(teacher_list):
            raise ValueError(
                'Student and teacher features do not have the same number '
                f'of tensors: {len(student_list)} vs {len(teacher_list)}.')

        losses = []
        for student_feat, teacher_feat in zip(student_list, teacher_list):
            teacher_feat = teacher_feat.detach()
            if self.feature_kd_loss_type == 'attention':
                student_repr = self._attention_map(student_feat)
                teacher_repr = self._attention_map(teacher_feat)
            else:
                student_repr = student_feat.reshape(student_feat.shape[0], -1)
                teacher_repr = teacher_feat.reshape(teacher_feat.shape[0], -1)

            if student_repr.shape != teacher_repr.shape:
                raise ValueError(
                    'Feature KD shape mismatch. Got student '
                    f'{tuple(student_repr.shape)} and teacher '
                    f'{tuple(teacher_repr.shape)}. Use '
                    "feature_kd_loss_type='attention' or matching backbones.")

            if self.feature_kd_loss_type in ('attention', 'mse'):
                loss = (student_repr - teacher_repr).pow(2).mean(dim=1)
            elif self.feature_kd_loss_type == 'l1':
                loss = (student_repr - teacher_repr).abs().mean(dim=1)
            elif self.feature_kd_loss_type == 'cosine':
                loss = 1 - F.cosine_similarity(
                    student_repr, teacher_repr, dim=1)
            else:
                raise ValueError(
                    f'Unsupported feature_kd_loss_type='
                    f'{self.feature_kd_loss_type!r}. Expected "attention", '
                    '"mse", "l1" or "cosine".')
            losses.append(loss)

        loss = torch.stack(losses, dim=0).mean(dim=0)
        # TSM features are (N * num_segments, ...): average over segments.
        if loss.numel() != batch_size:
            if loss.numel() % batch_size != 0:
                raise ValueError(
                    'Feature KD loss has an incompatible batch dimension: '
                    f'got {loss.numel()} values for batch_size={batch_size}.')
            loss = loss.reshape(batch_size, -1).mean(dim=1)
        return loss

    def compute_kd_sample_weights(
            self, teacher_logits: torch.Tensor) -> Optional[torch.Tensor]:
        if self.kd_sample_weight_mode in (None, 'none'):
            return None
        probs = F.softmax(
            teacher_logits.detach() / self.teacher_confidence_temperature,
            dim=1)
        return probs.max(dim=1).values.detach()

    @staticmethod
    def _weighted_mean(loss: torch.Tensor,
                       weights: Optional[torch.Tensor]) -> torch.Tensor:
        if weights is None:
            return loss.mean()
        weights = weights.to(device=loss.device, dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-12)

    def _scale_cls_loss(self, loss_cls: dict) -> dict:
        if self.cls_loss_weight != 1.0:
            for key, value in list(loss_cls.items()):
                if 'loss' in key:
                    loss_cls[key] = value * self.cls_loss_weight
        return loss_cls

    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        inputs_teacher = kwargs.pop('inputs_teacher', None)

        feats, loss_kwargs = self.extract_feat(
            inputs, data_samples=data_samples)
        loss_aux = loss_kwargs.get('loss_aux', dict())

        # No distillation term: plain student training.
        if self.kd_loss_weight <= 0 and self.feature_kd_loss_weight <= 0:
            loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)
            return merge_dict(self._scale_cls_loss(loss_cls), loss_aux)

        if inputs_teacher is None:
            raise ValueError(
                'inputs_teacher is required for cross-modal distillation')

        student_logits = self.cls_head(feats, **loss_kwargs)
        loss_cls = self._scale_cls_loss(
            self.cls_head.loss_by_feat(student_logits, data_samples))

        with torch.no_grad():
            teacher_feats, teacher_kwargs = self.teacher.extract_feat(
                inputs_teacher, data_samples=data_samples)
            teacher_logits = self.teacher.cls_head(teacher_feats,
                                                   **teacher_kwargs)

        kd_sample_weights = self.compute_kd_sample_weights(teacher_logits)
        loss_kd = self.compute_kd_loss(student_logits, teacher_logits)
        loss_cls['loss_kd'] = self._weighted_mean(
            loss_kd, kd_sample_weights) * self.kd_loss_weight

        if self.feature_kd_loss_weight > 0:
            loss_feature_kd = self.compute_feature_kd_loss(
                feats, teacher_feats, batch_size=student_logits.shape[0])
            feature_weights = (
                kd_sample_weights if self.apply_kd_weight_to_feature else None)
            loss_cls['loss_feature_kd'] = self._weighted_mean(
                loss_feature_kd, feature_weights) * self.feature_kd_loss_weight

        return merge_dict(loss_cls, loss_aux)

    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict student scores, plus teacher scores if RGB is given."""
        inputs_teacher = kwargs.pop('inputs_teacher', None)

        feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
        student_predictions = self.cls_head.predict(feats, data_samples,
                                                    **predict_kwargs)
        if inputs_teacher is None or self.teacher is None:
            return student_predictions

        teacher_predictions = self.teacher.predict(
            inputs_teacher, copy.deepcopy(data_samples))
        for student_pred, teacher_pred in zip(student_predictions,
                                              teacher_predictions):
            student_pred.set_field(
                {
                    'student': student_pred.pred_score,
                    'teacher': teacher_pred.pred_score
                }, 'pred_score')
        return student_predictions

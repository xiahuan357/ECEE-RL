import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    DeterministicPolicy,
    EnsembleContinuousQFunction,
    EnsembleQFunction,
    Policy,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api
from .base import TorchImplBase
from .utility import ContinuousQFunctionMixin
import Logger
import pandas as pd
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.linear_model import LassoCV
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




class DDPGBaseImpl(ContinuousQFunctionMixin, TorchImplBase, metaclass=ABCMeta):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleContinuousQFunction]
    _policy: Optional[Policy]
    _targ_q_func: Optional[EnsembleContinuousQFunction]
    _targ_policy: Optional[Policy]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._targ_policy = None
        self._actor_optim = None
        self._critic_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)

        # if self._use_gpu:
        #     self.to_gpu(self._use_gpu)
        # else:
        #     self.to_cpu()
        self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    @abstractmethod
    def _build_actor(self) -> None:
        pass

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # alternaste code
        # IBM
        cols = ['dob_mm', 'dob_wk', 'bfacil', 'ubfacil', 'bfacil3', 'mager41', 'mager14', 'mager9', 'restatus',
                'mbrace', 'mracerec', 'umhisp', 'mracehisp', 'mar', 'meduc', 'fagecomb', 'ufagecomb', 'fagerec11',
                'fbrace', 'fracerec', 'ufhisp', 'fracehisp', 'precare', 'precare_rec', 'uprevis', 'previs_rec',
                'wtgain', 'wtgain_rec', 'cig_1', 'cig_2', 'cig_3', 'rf_ncesar', 'urf_diab', 'urf_chyper', 'urf_phyper',
                'urf_eclam', 'uop_induc', 'uop_tocol', 'uld_meco', 'uld_precip', 'uld_breech', 'md_present', 'md_route',
                 'ume_vac', 'rdmeth_rec', 'dmeth_rec', 'attend', 'apgar5', 'apgar5r', 'dplural', 'dlmp_mm',
                'dlmp_yy', 'estgest', 'combgest', 'gestrec10', 'gestrec3', 'dbwt', 'bwtr14', 'bwtr4', 'uca_anen',
                'uca_spina', 'uca_ompha', 'uca_cleftlp', 'uca_downs', 'f_morigin', 'f_forigin', 'f_meduc', 'f_clinest',
                'f_apgar5', 'f_tobaco', 'f_rf_pdiab', 'f_rf_gdiab', 'f_rf_phyper', 'f_rf_ghyper', 'f_rf_eclamp',
                'f_rf_ppb', 'f_rf_ppo', 'f_rf_cesar', 'f_rf_ncesar',
                'f_ob_cervic', 'f_ob_toco', 'f_ob_succ', 'f_ob_fail', 'f_ol_rupture', 'f_ol_precip', 'f_ol_prolong',
                'f_ld_induct', 'f_ld_augment', 'f_ld_steroids', 'f_ld_antibio', 'f_ld_chorio', 'f_ld_mecon',
                'f_ld_fintol', 'f_ld_anesth', 'f_md_present', 'f_md_route', 'f_md_trial', 'f_ab_vent', 'f_ab_vent6',
                'f_ab_nicu', 'f_ab_surfac', 'f_ab_antibio', 'f_ab_seiz', 'f_ab_inj', 'f_ca_anen', 'f_ca_menin',
                'f_ca_heart', 'f_ca_hernia', 'f_ca_ompha', 'f_ca_gastro', 'f_ca_limb', 'f_ca_cleftlp', 'f_ca_cleft',
                'f_ca_downs', 'f_ca_chrom', 'f_ca_hypos', 'f_wtgain', 'f_mpcb', 'f_urf_diabetes', 'f_urf_chyper',
                'f_urf_phyper', 'f_urf_eclamp', 'f_uob_induct', 'f_uld_meconium', 'f_uld_precip',
                'f_uld_breech', 'f_u_forcep', 'f_u_vacuum', 'f_uca_anen', 'f_uca_spina', 'f_uca_omphalo',
                'f_uca_cleftlp', 'f_uca_downs', 'matchs', 'recwt', 'cig_rec', 'rf_diab', 'rf_gest', 'rf_phyp',
                'rf_ghyp', 'rf_eclam', 'rf_ppterm', 'rf_ppoutc', 'rf_cesar', 'op_cerv',
                'op_tocol', 'op_ecvs', 'op_ecvf', 'on_ruptr', 'on_abrup', 'on_prolg', 'ld_induct', 'ld_augment',
                'ld_steroids', 'ld_antibio', 'ld_chorio', 'ld_mecon', 'ld_fintol', 'ld_anesth', 'sex', 'ab_vent',
                'ab_vent6', 'ab_nicu', 'ab_surfac', 'ab_antibio', 'ca_anen', 'ca_menin', 'ca_heart', 'ca_hernia',
                'ca_ompha', 'ca_gastro', 'ca_limb', 'ca_cleftlp', 'ca_cleft', 'ca_hypos']

        D = pd.merge(pd.DataFrame(batch.observations.cpu().numpy(),
                                  columns=cols),
                     pd.DataFrame(batch.rewards.cpu().numpy(), columns=['rewards']),
                     left_index=True,
                     right_index=True)
        data = pd.merge(D, pd.DataFrame(batch.actions.cpu().numpy(), columns=['action']), left_index=True,
                        right_index=True)
        model = CausalModel(
            data=data,
            treatment='rewards',
            outcome='action',
            # common_causes=cols,

            graph='/path/to/data/ibm_x_482_x1.gml'
        )
        # end

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        propensity_strat_estimate = model.estimate_effect(identified_estimand,
                                                          method_name="backdoor.econml.metalearners.XLearner",
                                                          control_value=0,
                                                          treatment_value=1,
                                                          confidence_intervals=False,
                                                          method_params={"init_params": {
                                                               'models': GradientBoostingRegressor(),
                                                               'cate_models': GradientBoostingRegressor(),
                                                               'propensity_model': GradientBoostingClassifier()
                                                          },
                                                              "fit_params": {}}
                                                          )

        action_value = [propensity_strat_estimate.value]

        action = torch.tensor(action_value).reshape(1, 1, 1).to(torch.float32).to(device)


        rewards_uno = 1 - model.refute_estimate(identified_estimand,
                                                propensity_strat_estimate,show_progress_bar=True,
                                                method_name="random_common_cause",n_jobs=8,random_state=123).refutation_result['p_value']
        return self._q_func.compute_error(
            observations=batch.observations,
            # alternate code
            # actions=batch.actions,
            actions=action,
            # alternate code
            # rewards=batch.rewards,
            rewards=rewards_uno,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @abstractmethod
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)

    def update_actor_target(self) -> None:
        assert self._policy is not None
        assert self._targ_policy is not None
        soft_sync(self._targ_policy, self._policy, self._tau)

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._actor_optim
        return self._actor_optim

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._critic_optim
        return self._critic_optim


class DDPGImpl(DDPGBaseImpl):

    _policy: Optional[DeterministicPolicy]
    _targ_policy: Optional[DeterministicPolicy]

    def _build_actor(self) -> None:
        self._policy = create_deterministic_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        return -q_t.mean()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

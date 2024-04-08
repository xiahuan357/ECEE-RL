from convEncoder import ConvEncoderFactory

from d3rlpy.dataset import MDPDataset

# terminal interval
interval=250
# Number of experiments
exptimes = "5"

epochs=60

othername = 'xleaner_ReHid'

algos='SAC'

q_fun='mean'

scaler=0

shuffle=0

# -1 for all rows
data_lines=-1
# actor learn rate
aclr=0.01
# critic learn rate
crlr=0.01


import numpy as np


path="/root/PycharmProjects/paper/data/data_x_y.csv"
with open(path) as f:
    if data_lines== -1:
      data_x_y=np.loadtxt(path,delimiter=',',skiprows=1)
    else:
      data_x_y=np.loadtxt(path,delimiter=',',skiprows=1)[0:data_lines,:]
#Unable to read csv using Pandas and convert it to MDPdataset. There is a bug
path1="/root/PycharmProjects/paper/data/test_data_x_y.csv"
with open(path1) as f:
    if data_lines== -1:
       test_data_x_y=np.loadtxt(path1,delimiter=',',skiprows=1)
    else:
       test_data_x_y=np.loadtxt(path1,delimiter=',',skiprows=1)[0:int(data_lines/4),:]

if shuffle==1:

    indies = np.arange(len(data_x_y))
    np.random.shuffle(indies)
    data_x_y = data_x_y[indies]



cols = ['dob_mm','dob_wk','bfacil','ubfacil','bfacil3','mager41','mager14','mager9',
       'restatus','mbrace','mracerec','umhisp','mracehisp','mar','meduc','fagecomb',
       'ufagecomb','fagerec11','fbrace','fracerec','ufhisp','fracehisp','precare',
       'precare_rec','uprevis','previs_rec','wtgain','wtgain_rec','cig_1','cig_2',
        'cig_3','rf_ncesar','urf_diab','urf_chyper','urf_phyper','urf_eclam',
        'uop_induc','uop_tocol','uld_meco','uld_precip','uld_breech','md_present',
        'md_route','ume_vac','rdmeth_rec','dmeth_rec','attend','apgar5','apgar5r',
        'dplural','dlmp_mm','dlmp_yy','estgest','combgest','gestrec10','gestrec3',
        'dbwt','bwtr14','bwtr4','uca_anen','uca_spina','uca_ompha','uca_cleftlp',
        'uca_downs','f_morigin','f_forigin','f_meduc','f_clinest','f_apgar5','f_tobaco',
        'f_rf_pdiab','f_rf_gdiab','f_rf_phyper','f_rf_ghyper','f_rf_eclamp','f_rf_ppb',
        'f_rf_ppo','f_rf_cesar','f_rf_ncesar','f_ob_cervic',
        'f_ob_toco','f_ob_succ','f_ob_fail','f_ol_rupture','f_ol_precip',
        'f_ol_prolong','f_ld_induct','f_ld_augment','f_ld_steroids','f_ld_antibio',
        'f_ld_chorio','f_ld_mecon','f_ld_fintol','f_ld_anesth','f_md_present',
        'f_md_route','f_md_trial','f_ab_vent','f_ab_vent6','f_ab_nicu','f_ab_surfac',
        'f_ab_antibio','f_ab_seiz','f_ab_inj','f_ca_anen','f_ca_menin','f_ca_heart',
        'f_ca_hernia','f_ca_ompha','f_ca_gastro','f_ca_limb','f_ca_cleftlp','f_ca_cleft',
        'f_ca_downs','f_ca_chrom','f_ca_hypos','f_wtgain','f_mpcb','f_urf_diabetes',
        'f_urf_chyper','f_urf_phyper','f_urf_eclamp','f_uob_induct','f_uld_meconium',
        'f_uld_precip','f_uld_breech','f_u_forcep','f_u_vacuum','f_uca_anen','f_uca_spina',
        'f_uca_omphalo','f_uca_cleftlp','f_uca_downs','matchs','recwt','cig_rec','rf_diab',
        'rf_gest','rf_phyp','rf_ghyp','rf_eclam','rf_ppterm','rf_ppoutc','rf_cesar',
        'op_cerv','op_tocol','op_ecvs','op_ecvf','on_ruptr','on_abrup','on_prolg',
        'ld_induct','ld_augment','ld_steroids','ld_antibio','ld_chorio','ld_mecon',
        'ld_fintol','ld_anesth','sex','ab_vent','ab_vent6','ab_nicu','ab_surfac',
        'ab_antibio','ca_anen','ca_menin','ca_heart','ca_hernia','ca_ompha','ca_gastro',
        'ca_limb','ca_cleftlp','ca_cleft','ca_hypos']


observations =data_x_y[:,4:179]

#training set
actions = data_x_y[:,3].reshape(len(data_x_y),1)
rewards = data_x_y[:,2].reshape(len(data_x_y),1)
terminals=[]

for i in range(1,len(data_x_y)+1):
  if i%interval==0 or i==len(data_x_y):
    terminals.append(1)
  else:
    terminals.append(0)

dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)


#Test set
terminals_test=[]
observations_test = test_data_x_y[:,2:179]
actions_test = test_data_x_y[:,-2].reshape(len(test_data_x_y),1)
rewards_test = test_data_x_y[:,-1].reshape(len(test_data_x_y),1)

for i in range(1,len(test_data_x_y)+1):
  if i%interval ==0 or i==len(test_data_x_y):
    terminals_test.append(1)
  else:
    terminals_test.append(0)

dataset_test = MDPDataset(observations_test, actions_test, rewards_test, terminals_test, discrete_action=False)
import d3rlpy
d3rlpy.seed(123)
from d3rlpy.algos import AWAC,SAC,TD3
factory = ConvEncoderFactory(feature_size=cols.__len__())
from d3rlpy.models.q_functions import QRQFunctionFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory


if q_fun== 'mean':
    q_fun_name=MeanQFunctionFactory()
elif q_fun=='qr':
    q_fun_name=QRQFunctionFactory(n_quantiles=32)
else:
    print("Incorrect Q function configuration")

if scaler==1:
    scaler_name='min_max'
else:
    scaler_name=None

ALgo=None
if algos=='SAC':
    ALgo=SAC
elif algos=='TD3':
    ALgo=TD3
elif algos=='AWAC':
    ALgo=AWAC
else:
    print('Incorrect algorithm configuration')

td3= ALgo(actor_learning_rate=aclr,critic_learning_rate=crlr,
          use_gpu=True,scaler=scaler_name,reward_scaler=scaler_name,
          action_scaler=scaler_name,actor_encoder_factory=factory,
          critic_encoder_factory=factory,imitator_encoder_factory=factory,
          q_func_factory=q_fun_name)




from d3rlpy.metrics.scorer import continuous_action_diff_scorer_refute_dis,continuous_action_diff_scorer_real_dis,average_value_estimation_scorer



td3.fit(dataset,
        eval_episodes=dataset_test,
        n_epochs=epochs,
        tensorboard_dir='runs',
        experiment_name='ibm110_' + algos + '_nepos' + str(epochs) + '_ter' + str(
            interval) + '_SAMgra_' + exptimes + "_scaler" + str(scaler) + "_" + othername+'_'+q_fun+'_shuffle'+str(shuffle)+'_aclr'+str(aclr),
        scorers={
            'value_scale': average_value_estimation_scorer,
            'rl-action_diff': continuous_action_diff_scorer_refute_dis,
            'base_action_diff': continuous_action_diff_scorer_real_dis,
        },
        AddInfor='ibm110_' + algos + '_nepos' + str(epochs) + '_ter' + str(
            interval) + '_SAMgra_' + exptimes + "_scaler" + str(
            scaler) + "_" + othername + '_' + q_fun + '_shuffle' + str(shuffle) + '_aclr' + str(aclr),
        )









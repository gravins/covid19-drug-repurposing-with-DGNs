from DeepPurpose import DTI as models
from DeepPurpose import utils as dp_utils
from utils import compute_score
import ray
from torch import sigmoid
from torch import tensor

import warnings
warnings.filterwarnings("ignore")


@ray.remote(num_cpus=1, num_gpus=1/11) #10)
def train_ray(conf, conf_id, f_id, data, batch, epochs):

    tr_fold, vl_fold = data[f_id]

    train = dp_utils.data_process(tr_fold[0], tr_fold[1], tr_fold[2], 
                                    conf['drug_encoding'], conf['target_encoding'], 
                                    split_method='no_split')

    valid = dp_utils.data_process(vl_fold[0], vl_fold[1], vl_fold[2], 
                                    conf['drug_encoding'], conf['target_encoding'], 
                                    split_method='no_split')
    
    config = dp_utils.generate_config(
        drug_encoding = conf['drug_encoding'], 
        target_encoding = conf['target_encoding'], 
        batch_size = batch,
        train_epoch = epochs, 
        LR = conf['lr'],
        decay = 0.1, # similar to AdamW
        cnn_drug_filters = conf['cnn_drug_filters'] if 'cnn_drug_filters' in conf else None,
        cnn_drug_kernels = conf['cnn_drug_kernels'] if 'cnn_drug_kernels' in conf else None,
        cnn_target_filters = conf['cnn_target_filters'],
        cnn_target_kernels = conf['cnn_target_kernels'],
        cuda_id = None,
        gnn_hid_dim_drug = conf['gnn_hid_dim_drug'] if 'gnn_hid_dim_drug' in conf else None,
        gnn_num_layers = conf['gnn_num_layers'] if 'gnn_num_layers' in conf else None,
        gnn_activation = conf['gnn_activation'] if 'gnn_activation' in conf else None,        
        cls_hidden_dims = conf['cls_hidden_dims'] 
    )

    model = models.model_initialize(**config)
    model.train(train, valid, verbose=False)
    y_pred_tr = model.predict(train)
    y_pred_val = model.predict(valid)
    score_tr = compute_score((sigmoid(tensor(y_pred_tr)) > 0.5).float().tolist(), train['Label'].to_list())
    score_val = compute_score((sigmoid(tensor(y_pred_val)) > 0.5).float().tolist(), valid['Label'].to_list())

    return (f_id, conf_id, y_pred_tr, y_pred_val, score_tr, score_val)
 

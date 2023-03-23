
from random import choice 
import numpy as np 
import os, argparse, json

def write_one(dictionary_in, target):
    """
    write all key-value pairs of a dictionary to a json file
    """
    json.dump(dictionary_in, open(target, "w"))


def put_file_in_every_subfolder(folder, file):
    """
    put a file in every subfolder of a folder
    """
    for subfolder in os.listdir(folder):
        copy_file(file, os.path.join(folder, subfolder, file))

def copy_file(src, dst):
    """
    copy a file from src to dst
    """
    with open(src, "r") as f:
        content = f.read()  
    with open(dst, "w") as f:
        f.write(content)

def check_folder(folder):
    """
    check if folder exists, if not create it
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def generate_and_write(options):

    dictionary_categories = \
    {
        "categories": [3, 5],
        "category_weights_3": [[1.0, 1.5, 2.0], [1.0, 3.0, 4.0], [1.0, 5.0, 5.0]],
        "category_weights_5": [[2.0, 1.0, 2.0, 1.5, 1.0], [4.0, 1.0, 4.0, 3.0, 1.0], [5.0, 1.0, 5.0, 3.0, 1.0]]
    }

    dictionary_archi = \
    {
        "gated_num_layers": [1,2,3,4,5],
        "fc_num_layers": [1,2,3,4],
        "gated_hidden_size_1": [512, 1024, 2048],
        "gated_hidden_size_shape": ["flat", "cone"],
        "fc_hidden_size_1": [512, 1024],
        "fc_hidden_size_shape": ["flat", "cone"]
    }


    dictionary_values_options = \
    {
        "filter_outliers": [True],
        "filter_sparse_rxns": [False],
        "filter_species": [[3, 6]],
        "debug": [False],
        "test": [False],
        "batch_size": [256, 512],
        "embedding_size": [12, 16, 18, 20, 22],
        "epochs": [500,1000,1500],
        "fc_activation": ["ReLU"],
        "fc_batch_norm": [False],
        "freeze": [False, True],
        "gated_activation": ["ReLU"],
        "gated_num_fc_layers": [1, 2, 3, 4],
        "lr": [0.001],
        "dropout": [0.1, 0.25],
        "output_file": ["results.pkl"],
        "start_epoch": [0],
        "early_stop": [True],
        "scheduler": [False],
        "transfer_epochs": [250, 500, 1000],
        "transfer": [False, True],
        "freeze" : [True, False],
        "loss": ["huber"],
        "weight_decay": [0.0, 0.0001, 0.00001],
        "num_lstm_iters": [9, 11, 13, 15],
        "num_lstm_layers": [1, 2, 3],
        "gated_batch_norm": [0, 1],
        "gated_graph_norm":[0,1],
        "gated_residual":[True],
        "fc_batch_norm":[False, True]
    }   
    if(options["hydro"]):
        dictionary_values_options["augment"] = [False]
        dictionary_values_options["transfer"] = [False]
        dictionary_values_options["freeze"] = [False]

    else: 
        dictionary_values_options["augment"] = [False, True]
        dictionary_values_options["filter_sparse_rxns"] = [False]


    for i in range(options["num"]):

        dictionary_write = {}
        

        if(options["hydro"] == True or options["old_dataset"] == True):
            featurizer_dict = {
                #"choice_3":{
                #    "extra_features": ["bond_length", 'Lagrangian_K', 'e_density', 'lap_e_density', 
                #            'e_loc_func', 'ave_loc_ion_E', 'delta_g_promolecular', 'delta_g_hirsh', 'esp_nuc', 
                #            'esp_e', 'esp_total', 'grad_norm', 'lap_norm', 'eig_hess', 
                #            'det_hessian', 'ellip_e_dens', 'eta'],
                #    "feature_filter": True
                #},
                #"choice_4":{
                #    "extra_features": ["bond_length", 'esp_total', 'Lagrangian_K', 'ellip_e_dens'], 
                #    "feature_filter": True
                #},
                "choice_5":{
                    "extra_features": ['esp_total'], 
                    "feature_filter": True
                },
                "choice_6":{
                    "extra_features": ["bond_length"], 
                    "feature_filter": True
                },
                #"choice_7":{
                #    "feature_filter": False
                #},
            }

        else: 
            featurizer_dict = {
                '''
                "choice_1":{
                    "extra_features": [
                        "1_s", "2_s", "1_p", "2_p", "1_d", "2_d", "1_f", "2_f", 
                        "1_polar", "2_polar", "occ_nbo", "valence_electrons", "total_electrons", 
                        "partial_charges_nbo", "partial_charges_mulliken", 
                        "partial_charges_resp", "indices_nbo"], 
                    "feature_filter": True
                },
                "choice_2":{
                    "extra_features": [       
                        "esp_nuc", "esp_e", "esp_total",
                        "ellip_e_dens", "indices_qtaim"
                        ],
                    "feature_filter": True
                },
                "choice_3": {
                        "extra_features": [     
                        "esp_nuc", "esp_e", "esp_total", "ellip_e_dens", "bond_length", 
                        ],
                    "feature_filter": True
                }, 

                "choice_4":{
                    "extra_features": [
                        "bond_length", "1_s", "2_s", "1_p", "2_p", "1_d", "2_d", "1_f", "2_f", 
                        "1_polar", "2_polar", "occ_nbo", "valence_electrons", "total_electrons", 
                        "partial_charges_nbo", "partial_charges_mulliken", "partial_charges_resp",
                           "indices_nbo"], 
                    "feature_filter": True
                },
                '''
                "choice_5":{
                    "extra_features": ["bond_length"], 
                    "feature_filter": False
                }
            }
        featurizer_settings = choice(list(featurizer_dict.keys()))
        featurizer_settings = featurizer_dict[featurizer_settings]
        dictionary_write.update(featurizer_settings)

        if(options["class_cats"] == 3):
            dictionary_write["categories"] = 3
            dictionary_write["category_weights"] = choice(dictionary_categories["category_weights_3"])
        else: 
            dictionary_write["categories"] = 5
            dictionary_write["category_weights"] = choice(dictionary_categories["category_weights_5"])
            
        base_fc = choice(dictionary_archi["gated_hidden_size_1"])
        base_gat = choice(dictionary_archi["fc_hidden_size_1"])
        shape_fc = choice(dictionary_archi["fc_hidden_size_shape"])
        shape_gat = choice(dictionary_archi["gated_hidden_size_shape"])

        if(shape_fc == "flat"):
            fc_layers = [base_fc for i in range(choice(dictionary_archi["fc_num_layers"]))]
        else:
            fc_layers = [int(base_fc/(2**i)) for i in range(choice(dictionary_archi["fc_num_layers"]))]

        if(shape_gat == "flat"):
            gat_layers = [base_gat for i in range(choice(dictionary_archi["gated_num_layers"]))]
        else:
            gat_layers = [int(base_gat/(2**i)) for i in range(choice(dictionary_archi["gated_num_layers"]))]


        dictionary_write["fc_num_layers"] = len(fc_layers)
        dictionary_write["gated_num_layers"] = len(gat_layers)
        dictionary_write["gated_hidden_size"] = gat_layers
        dictionary_write["fc_hidden_size"] = fc_layers
        dictionary_write["dataset_loc"] = options["dataset_loc"]
        dictionary_write["on_gpu"] = options["gpu"]
        dictionary_write["classifier"] = options["classifier"]
        dictionary_write["restore"] = True
        
        for k, v in dictionary_values_options.items():
            dictionary_write[k] = choice(v)

        if(options["hydro"]):
            folder = "../hydro_training"

        elif(options["old_dataset"]):
            folder = "../old_mg_training"
        
        else: 
            folder = "../mg_training"

        if(options["gpu"]): folder += "_gpu"
        else: folder += "_cpu"

        if(options["classifier"]):
            folder += "_classifier/"
        else:
            folder += "_regressor/"

        check_folder(folder)

        if(options["gpu"]): target = folder + "gpu_"
        else: target = folder + "cpu_"

    
        target += str(int(np.floor(i / options["per_folder"])))
        check_folder(target)
        target += "/settings" + str(i) + ".json"
        
        write_one(dictionary_write, target)


    if(options["perlmutter"] == False): 
        if(options["gpu"]):
            slurm_file = "./xsede_gpu_lightning.sh"
            #if(options["hydro"]):
            #    slurm_file = "xsede_gpu_hydro_lightning.sh"
        else: 
            slurm_file = "./xsede_cpu_lightning.sh"
            #if(options["hydro"]): 
            #    slurm_file = "./xsede_cpu_hydro_lightning.sh"

    else: 
        if(options["gpu"]):
            slurm_file = "./perlmutter_gpu_lightning.sh"
            #if(options["hydro"]):
            #    slurm_file = "./perlmutter_gpu_hydro.sh"
        else: 
            slurm_file = "./perlmutter_cpu_lightning.sh"
            #if(options["hydro"]):
            #    slurm_file = "./perlmutter_cpu_hydro.sh"
        
    put_file_in_every_subfolder(folder, slurm_file)

def main():
    # create argparse 
    parser = argparse.ArgumentParser(description='Create settings files for training')
    parser.add_argument('--perlmutter', action='store_true', help='Use perlmutter')
    parser.add_argument('--gpu', action='store_true', help='Use gpu')
    parser.add_argument('--hydro', action='store_true', help='Use hydro')
    parser.add_argument('--old_dataset', action='store_true', help='Use old dataset')
    parser.add_argument('--imputed_data', action='store_true', help='Use imputed data in training')
    parser.add_argument('--classifier', action='store_true', help='Use classifier')
    parser.add_argument('--class_cats', type=int, default=3, help='Number of categories')
    # number of runs 
    parser.add_argument('--num', type=int, default=50, help='Number of runs')
    # number of runs per folder
    parser.add_argument('--per_folder', type=int, default=50, help='Number of runs per folder')
    # parse arguments
    args = parser.parse_args()
    options = vars(args)

    classifier = options["classifier"]
    class_cats = options["class_cats"]
    gpu = options["gpu"]
    hydro = options["hydro"]
    perlmutter = options["perlmutter"]
    old_dataset = options["old_dataset"]
    num = options["num"]
    per_folder = options["per_folder"]
    imputed = options["imputed_data"]
    
    if(classifier and class_cats != 3 and class_cats != 5):
        raise ValueError("Must have 3 or 5 categories for classifier")

    if hydro:
        dataset_loc = "../../../../dataset/qm_9_merge_3_qtaim.json"
    
    elif old_dataset: 
        dataset_loc = "../../../../dataset/mg_dataset/merged_mg.json"
    
    else: 
        if not imputed:
            dataset_loc = "../../../../dataset/mg_dataset/mg_qtaim_complete_nonimputed.json"
        else:
            dataset_loc = "../../../../dataset/mg_dataset/mg_qtaim_complete.json"

    options_dict = {
        "dataset_loc": dataset_loc,
        "classifier": classifier,
        "class_cats": class_cats, 
        "hydro": hydro, 
        "old_dataset": old_dataset,
        "num": num,  
        "per_folder": per_folder,
        "gpu": gpu,
        "perlmutter": perlmutter
    }

    generate_and_write(options_dict)

    
main()

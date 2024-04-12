import torch
import dgl
import itertools
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np 
import torch.autograd.profiler as profiler
import time

# import threading
#! each worker is responsible for one minibatch. 
#! so each worker will be called samples/batch_size times.
# worker_info = threading.local()

class DataLoaderReaction(DataLoader):
    """
    This dataloader works specifically for the reaction dataset where each reaction is
    represented by a list of the molecules (i.e. reactants and products).

    Also, the label value of each datapoint should be of the same shape.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            # reaction_graph, reaction_features, labels = map(list, zip(*samples))  # old

            reactions, labels = map(list, zip(*samples))
            molecules = self.dataset.graphs


            mol_ids = set()
            for rxn in reactions:
                mol_ids.update(rxn.init_reactants + rxn.init_products)
            mol_ids = sorted(mol_ids)
            global_to_subset_mapping = {g: s for s, g in enumerate(mol_ids)}

            for rxn in reactions:
                rxn.reactants = [global_to_subset_mapping[i] for i in rxn.init_reactants]
                rxn.products = [global_to_subset_mapping[i] for i in rxn.init_products]

            # molecules subset
            graphs = [molecules[i] for i in mol_ids]


            batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]

            #reaction_types = [la["reaction_type"] for la in labels]

            batched_labels = {
                "value": target,
                "value_rev": value_rev,
                "reaction": reactions,
                "id": identifier,
                #"reaction_types": reaction_types,
            }


            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            # graph norm
            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderReaction, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderReactionLMDB(DataLoader):
    """
    This dataloader works specifically for the reaction network where a the reactions
    are constructed from a list of reactions.
    """

    def __init__(self, dataset, **kwargs):
        
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            # print("samples",len(samples))
            start_time = time.time()
            #! 1. mol_graph, reactions, labels
            reactions, labels = map(list, zip(*samples))
            graphs = self.dataset.graphs

            #! 2. subselect mol_graph and batch graphs
            mol_ids = set()
            for rxn in reactions:
                mol_ids.update(
                    rxn["reaction_molecule_info"]["reactants"]["init_reactants"]
                    + rxn["reaction_molecule_info"]["products"]["init_products"]
                )
            mol_ids = sorted(mol_ids)
            global_to_subset_mapping = {g: s for s, g in enumerate(mol_ids)}
            ################### from reasction network ######################

            #global_to_subset_mapping = {g: s for s, g in enumerate(mol_id_map)} 
            #graphs_unsorted = list(itertools.chain.from_iterable(graphs))
            #graphs_unsorted = []
            #for reaction_graphs in graphs:
            #    [graphs_unsorted.append(graph_temp) for graph_temp in reaction_graphs[0]]
            #    [graphs_unsorted.append(graph_temp) for graph_temp in reaction_graphs[1]]
            #print(len(graphs_unsorted), len(mol_id_map))
            #print(mol_id_map)
            #graphs_unsorted = []
            #graphs_sorted = []
            #test_sort = []

            #!3. change global-local mapping in rxn
            for rxn in reactions:
                init_reactants = rxn["reaction_molecule_info"]["reactants"][
                    "init_reactants"
                ]
                init_products = rxn["reaction_molecule_info"]["products"]["init_products"]
                mapped_reactants = [global_to_subset_mapping[i] for i in init_reactants]
                mapped_products = [global_to_subset_mapping[i] for i in init_products]
                rxn["reaction_molecule_info"]["reactants"]["reactants"] = mapped_reactants
                rxn["reaction_molecule_info"]["products"]["products"] = mapped_products

            # molecules subset
            sub_molecules = [graphs[i] for i in mol_ids]
            
            #print(mol_id_map)
            #print(test_sort)
            #graphs = graphs_sorted    
            # get graphs, ordered by mol_ids
            
            # resort graphs to match the new mol_ids
            batched_graphs = dgl.batch(sub_molecules)
            ################### from reaction network ######################
            
            #! 4. process labels
            #batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in sub_molecules]
            sizes_bond = [g.number_of_nodes("bond") for g in sub_molecules]

            target = torch.stack([la["value"] for la in labels])
            value_rev = torch.stack([la["value_rev"] for la in labels])
            identifier = [la["id"] for la in labels]


            batched_labels = {
                "value": target,
                "value_rev": value_rev,
            #!    "reaction": reactions, #!wx
                "id": identifier
            }

            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                try: #Can be fixed
                    #*if it is a folder
                    batched_labels["scaler_stdev"] = torch.stack(stdev)
                except:
                    #* if it is a single file
                    batched_labels["scaler_stdev"] = torch.FloatTensor(stdev)
                
            except KeyError:
                pass

            # graph norm
            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            #!5. calculate batch_indices and return a dict
            atom_batch_indices = get_node_batch_indices(batched_graphs, "atom")
            bond_batch_indices = get_node_batch_indices(batched_graphs, "bond")
            global_batch_indices = get_node_batch_indices(batched_graphs, "global")
            

            # print("SHAPE", atom_batch_indices.shape, bond_batch_indices.shape, global_batch_indices.shape)
            device = target.device
            #!batch reaction and features
            (batched_rxn_graphs, 
            batched_atom_reactant, 
            batched_atom_product, 
            batched_bond_reactant, 
            batched_bond_product,
            batched_global_reactant,
            batched_global_product,
            global_batch_indices_reactant,
            global_batch_indices_product
            )=create_batched_reaction_data(reactions, 
                                atom_batch_indices,
                                bond_batch_indices,
                                global_batch_indices, 
                                device)


            batched_indices_dict = {
            "atom_batch_indices":atom_batch_indices,
            "bond_batch_indices":bond_batch_indices,
            "global_batch_indices":global_batch_indices,
            "batched_rxn_graphs" : batched_rxn_graphs, 
            "batched_atom_reactant": batched_atom_reactant, 
            "batched_atom_product": batched_atom_product, 
            "batched_bond_reactant": batched_bond_reactant, 
            "batched_bond_product":batched_bond_product,
            "batched_global_reactant":batched_global_reactant,
            "batched_global_product":batched_global_product,
            "global_batch_indices_reactant":global_batch_indices_reactant,
            "global_batch_indices_product":global_batch_indices_product
            }


            # worker_id = getattr(worker_info, 'id', None)
            # print("Worker_id:", worker_id ,"TIME:", time.time() - start_time)

            return batched_graphs, batched_labels, batched_indices_dict
        
        super(DataLoaderReactionLMDB, self).__init__(dataset, collate_fn=collate, **kwargs)

        # super(DataLoaderReactionLMDB, self).__init__(dataset, collate_fn=collate, worker_init_fn=worker_init_fn, **kwargs)


# def worker_init_fn(worker_id):
#     # Store the worker id in thread-local storage
#     worker_info.id = worker_id

def get_node_batch_indices(batched_graph, node_type):
    """
    Generate batch indices for each node of the specified type in a batched DGL graph.
    
    Args:
    - batched_graph (DGLGraph): The batched graph.
    - node_type (str): The type of nodes for which to generate batch indices.
    
    Returns:
    - torch.Tensor: The batch indices for each node of the specified type.
    """
    batch_num_nodes = batched_graph.batch_num_nodes(node_type)
    return torch.repeat_interleave(torch.arange(len(batch_num_nodes), device=batched_graph.device), batch_num_nodes)


def get_batch_indices_mapping(batch_indices, reactant_ids, atom_bond_map, atom_bond_num, device):
    distinguishable_value = torch.iinfo(torch.long).max
    indices_full = torch.full((atom_bond_num,), distinguishable_value, dtype=torch.long, device=device)
    sorted_index_reaction = [torch.tensor([value for key, value in sorted(d.items())], device=device) for d in atom_bond_map]

    # print("batch_indices", batch_indices)
    # print("reactant_ids", reactant_ids)

    matches = torch.tensor([idx for rid in reactant_ids for idx in (batch_indices == rid).nonzero(as_tuple=True)[0]], device=device)

    sorted_values_concat = torch.cat(sorted_index_reaction)


    # print("values_concat:",len(sorted_values_concat), "matches:", matches)
    indices_full[sorted_values_concat] = matches
    return indices_full

def create_batched_reaction_data(reactions,atom_batch_indices,
                                 bond_batch_indices, global_batch_indices, device):
    
    batched_graphs = dgl.batch([reaction['reaction_graph'] for reaction in reactions])


    batched_atom_reactant = torch.tensor([],dtype=torch.long, device=device) #torch.Tensor([], device=device)
    batched_atom_product =  torch.tensor([], dtype=torch.long, device=device)#torch.Tensor([], device=device)
    batched_bond_reactant = torch.tensor([], dtype=torch.long, device=device) #torch.Tensor([], device=device)
    batched_bond_product = torch.tensor([], dtype=torch.long, device=device) # torch.Tensor([], device=device)
    batched_global_reactant = torch.tensor([], dtype=torch.long, device=device) 
    batched_global_product = torch.tensor([], dtype=torch.long, device=device)  

    global_batch_indices_reactant = torch.tensor([], dtype=torch.long, device=device) 
    global_batch_indices_product = torch.tensor([], dtype=torch.long, device=device)  


    # idx = 0
    for idx, reaction in enumerate(reactions):
            #print("idx:", idx)
            #breakpoint()
            # print(">>>>>>>>>>>>>>>>id:", idx)
            # idx+=1
            num_atoms_total = reaction["mappings"]["num_atoms_total"]
            num_bond_total = reaction["mappings"]["num_bonds_total"]
            #!reactant
            #batched_indices_reaction for reactant.
            reactant_ids= reaction["reaction_molecule_info"]["reactants"]["reactants"]
            reactant_ids = torch.tensor(reactant_ids, device=device)
            #!global
            batched_global_reactant = torch.cat((batched_global_reactant, reactant_ids),dim=0)
            #breakpoint()
            global_batch_indices_reactant=torch.cat((global_batch_indices_reactant, torch.tensor([idx]*len(reactant_ids), dtype=torch.long, device=device)),dim=0)
            #!atom
            atom_map_react = reaction["mappings"]["atom_map"][0]
            batch_indices_react=get_batch_indices_mapping(atom_batch_indices, reactant_ids, atom_map_react, num_atoms_total, device=device)
            #batched_atom_reactant.extend(batch_indices_react)
            batched_atom_reactant = torch.cat((batched_atom_reactant, batch_indices_react),dim=0)
            #!bond
            #breakpoint()
            bond_map_react = reaction["mappings"]["bond_map"][0]
            batch_indices_react=get_batch_indices_mapping(bond_batch_indices, reactant_ids, bond_map_react, num_bond_total, device=device)
            #batched_bond_reactant.extend(batch_indices_react)
            batched_bond_reactant = torch.cat((batched_bond_reactant, batch_indices_react),dim=0)
            #!product
            #batched_indices_reaction for product.
            product_ids= reaction["reaction_molecule_info"]["products"]["products"]
            product_ids = torch.tensor(product_ids, device=device)
            #!global 
            batched_global_product = torch.cat((batched_global_product, product_ids),dim=0)
            global_batch_indices_product=torch.cat((global_batch_indices_product, torch.tensor([idx]*len(product_ids), dtype=torch.long, device=device)),dim=0)
            #!atom
            atom_map_product = reaction["mappings"]["atom_map"][1]
            batch_indices_product=get_batch_indices_mapping(atom_batch_indices, product_ids, atom_map_product, num_atoms_total, device=device)
            #batched_atom_product.extend(batch_indices_product)
            batched_atom_product = torch.cat((batched_atom_product, batch_indices_product), dim=0)
            #!bond
            bond_map_product = reaction["mappings"]["bond_map"][1]
            batch_indices_product=get_batch_indices_mapping(bond_batch_indices, product_ids, bond_map_product, num_bond_total, device=device)
            #batched_bond_product.extend(batch_indices_product)
            batched_bond_product = torch.cat((batched_bond_product, batch_indices_product), dim=0)
    
    #!batched indices will be used after MP step.
    return batched_graphs, batched_atom_reactant, batched_atom_product, batched_bond_reactant, batched_bond_product, batched_global_reactant, batched_global_product, global_batch_indices_reactant, global_batch_indices_product
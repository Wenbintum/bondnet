import itertools
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict, OrderedDict
from bondnet.core.reaction import ReactionsMultiplePerBond, ReactionsOnePerBond
from bondnet.utils import create_directory, pickle_load, pickle_dump, yaml_dump, to_path

logger = logging.getLogger(__name__)

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class ReactionCollection:
    """
    A list of Reactions, and operations on them.
    """

    def __init__(self, reactions):
        """
        Args:
            reactions (list): a sequence of :class:`Reaction`.
        """
        self.reactions = reactions

    @classmethod
    def from_file(cls, filename):
        d = pickle_load(filename)
        logger.info(
            "{} reactions loaded from file: {}".format(len(d["reactions"]), filename)
        )
        return cls(d["reactions"])

    def to_file(self, filename="reactions.pkl"):
        logger.info("Start writing reactions to file: {}".format(filename))

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactions": self.reactions,
        }
        pickle_dump(d, filename)

    def get_counts_by_broken_bond_type(self):
        """
        Count the reactions by broken bond type (species of atoms forming the bond).

        Returns:
            dict: {(spec1, spec2): counts} where (spec1, spec2) denotes the type of the
                bond and counts is its number if the collection.
        """
        counts = defaultdict(int)
        for rxn in self.reactions:
            bond_type = tuple(sorted(rxn.get_broken_bond_attr()["species"]))
            counts[bond_type] += 1
        return counts

    def get_counts_by_reactant_charge(self):
        """
        Count the reactions by the reactant charge.

        Returns:
            dict: {charge: counts} both charge and counts are integers.
        """
        counts = defaultdict(int)
        for rxn in self.reactions:
            charge = rxn.reactants[0].charge
            counts[charge] += 1
        return counts

    def get_counts_by_reaction_charge(self):
        """
        Count the reactions by the charge of the reactant and the products.

        Reaction charge means the charges of all the reactants and the products.

        Returns:
            list of dict: each dict gives the counts of specific combination of
            reactants charge and products charge: { "reactants_charge": [],
            "products_charge":[], "counts": counts}
        """
        counts = defaultdict(int)
        for rxn in self.reactions:
            rcts_charge = tuple(sorted([m.charge for m in rxn.reactants]))
            prdts_charge = tuple(sorted([m.charge for m in rxn.products]))
            charges = (rcts_charge, prdts_charge)
            counts[charges] += 1

        counts_list = []
        for (rcts_charge, prdts_charge), cts in counts.items():
            counts_list.append(
                {
                    "reactants_charge": rcts_charge,
                    "products_charge": prdts_charge,
                    "counts": cts,
                }
            )

        return counts_list

    def filter_by_bond_type(self, bond_type):
        """
        Filter the reactions by the type (species of the atoms forming the bond) of the
        breaking bond.

        Args:
            bond_type (tuple of string): species of the two atoms forming the bond
        """
        new_rxns = []
        for rxn in self.reactions:
            attr = rxn.get_broken_bond_attr()
            species = attr["species"]
            if set(species) == set(bond_type):
                new_rxns.append(rxn)
        self.reactions = new_rxns

    def filter_by_reactant_charge(self, charge):
        """
        Filter reactions by charge of reactant.

        Args:
            charge (int): charge of reactant
        """
        new_rxns = []
        for rxn in self.reactions:
            if rxn.reactants[0].charge == charge:
                new_rxns.append(rxn)
        self.reactions = new_rxns

    def filter_by_reactant_and_product_charge(self, reactants_charge, products_charge):
        """
        Filter reactions by the charge of the reactants and the products.

        For the retained reactions, the number of reactants equals to the length of
        `reactants_charge` and similarly the number of products equals to the
        length of `products_charge`.

        Args:
            reactants_charge (list of int): charge of reactants
            products_charge (list of int): charge of products
        """
        n_rct = len(reactants_charge)
        n_prdt = len(products_charge)
        charge_rct = set(reactants_charge)
        charge_prdt = set(products_charge)

        new_rxns = []
        for rxn in self.reactions:
            if (
                len(rxn.reactants) == n_rct
                and len(rxn.products) == n_prdt
                and set([m.charge for m in rxn.reactants]) == charge_rct
                and set([m.charge for m in rxn.products]) == charge_prdt
            ):
                new_rxns.append(rxn)

        self.reactions = new_rxns

    def plot_heatmap_of_counts_by_broken_bond_type(
        self, filename="heatmap_bond_type.pdf", title=None, **kwargs
    ):
        """
        Plot a heatmap of the counts of reactions by broken type (species of the two
        atoms forming the bond).

        Args:
            filename (Path): path of the to-be-generated plot.
            title (str): title for the plot.
            kwargs: keyword arguments for matplotlib.imshow()
        """

        def plot_heatmap(matrix, labels):
            fig, ax = plt.subplots()
            im = ax.imshow(matrix, vmin=np.min(matrix), vmax=np.max(matrix), **kwargs)
            plt.colorbar(im)

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(labels)), minor=False)
            ax.set_yticks(np.arange(len(labels)), minor=False)

            # label them with the respective list entries
            ax.set_xticklabels(labels, minor=False)
            ax.set_yticklabels(labels, minor=False)
            ax.set_xlim(-0.5, len(labels) - 0.5)
            ax.set_ylim(len(labels) - 0.5, -0.5)

            if title is not None:
                ax.set_title(title)

            fig.savefig(to_path(filename), bbox_inches="tight")

        def prepare_data(counts_by_bond_type):
            species = set()
            for bond_type in counts_by_bond_type:
                species.update(bond_type)
            species = sorted(species)
            data = np.zeros((len(species), len(species))).astype(np.int32)
            for s1, s2 in itertools.combinations_with_replacement(species, 2):
                idx1 = species.index(s1)
                idx2 = species.index(s2)
                key = tuple(sorted([s1, s2]))
                data[idx1, idx2] = data[idx2, idx1] = counts_by_bond_type[key]
            return data, species

        counts = self.get_counts_by_broken_bond_type()
        data, species = prepare_data(counts)
        plot_heatmap(data, species)

    def plot_bar_of_counts_by_reactant_charge(
        self, filename="barplot_counts_by_reactant_charge.pdf", title=None, **kwargs
    ):
        """
        Create a bar plot for the counts of reactions by reactant charge.

        Args:
            filename (Path): path of the to-be-generated plot.
            title (str): title for the plot.
            kwargs: keyword arguments for matplotlib.bar()
        """

        def plot_bar(counts, labels):
            fig, ax = plt.subplots()
            ax.grid(axis="y", zorder=1)

            X = np.arange(len(counts))
            ax.bar(X, counts, tick_label=labels, zorder=2, **kwargs)

            ax.set_xlabel("Reactant charge")
            ax.set_ylabel("Counts")

            if title is not None:
                ax.set_title(title)

            fig.savefig(filename, bbox_inches="tight")

        counts_dict = self.get_counts_by_reactant_charge()
        labels = sorted(counts_dict.keys())
        counts = [counts_dict[k] for k in labels]

        plot_bar(counts, labels)

    def plot_bar_of_counts_by_reaction_charge(
        self, filename="barplot_counts_by_reaction_charge.pdf", title=None, **kwargs
    ):
        """
        Create a bar plot for the counts of reactions by reaction charge.

        Reaction charge means the charges of all the reactants and the products.

        Args:
            filename (Path): path of the to-be-generated plot.
            title (str): title for the plot.
            kwargs: keyword arguments for matplotlib.bar()
        """

        def plot_bar(counts, labels):
            fig, ax = plt.subplots()
            ax.grid(axis="y", zorder=1)

            X = np.arange(len(counts))
            ax.bar(X, counts, tick_label=labels, zorder=2, **kwargs)

            ax.set_xticklabels(labels, rotation=45, ha="right")

            ax.set_xlabel("[(reactants charge), (products charge)]")
            ax.set_ylabel("Counts")

            if title is not None:
                ax.set_title(title)

            fig.savefig(filename, bbox_inches="tight")

        counts_dict = self.get_counts_by_reaction_charge()
        # group by number of products
        one_product = []
        two_products = []
        others = []
        for d in counts_dict:
            if len(d["products_charge"]) == 1:
                one_product.append(d)
            elif len(d["products_charge"]) == 2:
                two_products.append(d)
            else:
                others.append(d)
        # sort by reactant charge, assuming only one reactant
        one_product = sorted(one_product, key=lambda d: d["reactants_charge"][0])
        two_products = sorted(two_products, key=lambda d: d["reactants_charge"][0])
        others = sorted(others, key=lambda d: d["reactants_charge"][0])

        # get labels and counts
        labels = []
        counts = []
        for d in one_product:
            labels.append(str([d["reactants_charge"], d["products_charge"]]))
            counts.append(d["counts"])
        for d in two_products:
            labels.append(str([d["reactants_charge"], d["products_charge"]]))
            counts.append(d["counts"])
        for d in others:
            labels.append(str([d["reactants_charge"], d["products_charge"]]))
            counts.append(d["counts"])

        plot_bar(counts, labels)

    def plot_histogram_of_reaction_energy(
        self, filename="histogram_reaction_energy.pdf", title=None, **kwargs
    ):
        """
        Plot histogram of the reaction energy.

        Note, this will use all reactions in the reaction collection. If you want to
        plot for a specific bond type or charge, filter the reaction first.

        Args:
            filename (Path): path of the to-be-generated plot.
            title (str): title for the plot.
            kwargs: keyword arguments for matplotlib.hist()
        """

        def plot_hist(data, xmin, xmax):
            fig, ax = plt.subplots()
            ax.hist(data, 20, range=(xmin, xmax), **kwargs)

            ax.set_xlim(xmin, xmax)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Counts")

            if title is not None:
                ax.set_title(title)

            fig.savefig(to_path(filename), bbox_inches="tight")

        energies = []
        for rxn in self.reactions:
            energies.append(rxn.get_free_energy())
        xmin = min(energies) - 0.5
        xmax = max(energies) + 0.5

        plot_hist(energies, xmin, xmax)

    def plot_histogram_of_broken_bond_length(
        self, filename="histogram_broken_bond_length.pdf", title=None, **kwargs
    ):
        """
        Plot histogram of the length of the broken bond.

        Args:
            filename (Path): path of the to-be-generated plot.
            title (str): title for the plot.
            kwargs: keyword arguments for matplotlib.hist()
        """

        def plot_hist(data):
            fig, ax = plt.subplots()
            ax.hist(data, 20, **kwargs)

            ax.set_xlabel("Bond length")
            ax.set_ylabel("Counts")

            if title is not None:
                ax.set_title(title)

            fig.savefig(to_path(filename), bbox_inches="tight")

        bond_lengths = []
        for rxn in self.reactions:
            coords = rxn.reactants[0].coords
            u, v = rxn.get_broken_bond()
            d = np.linalg.norm(coords[u] - coords[v])
            bond_lengths.append(d)

        plot_hist(bond_lengths)

    def group_by_reactant(self):
        """
        Group reactions that have the same reactant together.

        Returns:
            dict: with reactant as the key and list of reactions as the value
        """
        grouped_reactions = defaultdict(list)
        for rxn in self.reactions:
            reactant = rxn.reactants[0]
            grouped_reactions[reactant].append(rxn)
        # print("grouped reaction by reactant: "+ str(len(grouped_reactions)))
        return grouped_reactions

    def group_by_reactant_charge_0(self):
        """
        Group reactions that have the same reactant together, keeping charge 0
        reactions (charges of reactant and products are all 0).

        A group of reactions of the same reactant are put in to
        :class:`ReactionsOnePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsOnePerBond`
        """
        groups = self.group_by_reactant()

        new_groups = []
        zero_charge_rxns = []

        for reactant in groups:
            for rxn in groups[reactant]:
                zero_charge = True
                # for m in rxn.reactants + rxn.products:
                #    #if m.charge != 0:
                #    #    zero_charge = False
                #    #    break
                if zero_charge:
                    zero_charge_rxns.append(rxn)
            # add to new group only when at least has one reaction
            if zero_charge_rxns:
                ropb = ReactionsOnePerBond(reactant, zero_charge_rxns)
                new_groups.append(ropb)
        return new_groups

    def group_by_reactant_lowest_energy(self):
        """
        Group reactions that have the same reactant together.

        For reactions that have the same reactant and breaks the same bond, we keep the
        reaction that have the lowest energy across products charge.

        A group of reactions of the same reactant are put in to
        :class:`ReactionsOnePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsOnePerBond`
        """

        groups = self.group_by_reactant()
        # print("groups by reactant" + str(len(groups)))

        new_groups = []
        for reactant in groups:
            # find the lowest energy reaction for each bond
            lowest_energy_reaction = dict()
            for rxn in groups[reactant]:
                bond = rxn.get_broken_bond()
                if bond not in lowest_energy_reaction:
                    lowest_energy_reaction[bond] = rxn
                else:
                    e_old = lowest_energy_reaction[bond].get_free_energy()
                    e_new = rxn.get_free_energy()
                    if e_new < e_old:
                        lowest_energy_reaction[bond] = rxn

            ropb = ReactionsOnePerBond(reactant, lowest_energy_reaction.keys())
            new_groups.append(ropb)
        # print("reactants groups again or whatever" + str(len(new_groups)))
        return new_groups

    def group_by_reactant_all(self):
        """
        Group reactions that have the same reactant together.

        A group of reactions of the same reactant are put in to
        :class:`ReactionsMultiplePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsMultiplePerBond`
        """

        groups = self.group_by_reactant()
        new_groups = [
            ReactionsMultiplePerBond(reactant, rxns)
            for reactant, rxns in groups.items()
        ]

        return new_groups


    def calculate_broken_bond_fraction(self):
        """
        For each unique reactant, calculate the fraction of bonds that are broken with
        respect to all the bonds.

        Note, this requires that, when extracting the reactions, all the reactions
        of a reactant (ignoring symmetry) are extracted, i.e. set `find_one` to False
        in :method:`ReactionExtractorFromMolSet.extract_one_bond_break`.

        Returns:


        """
        groups = self.group_by_reactant()

        num_bonds = []
        frac = []
        for reactant, rxns in groups.items():
            rmb = ReactionsMultiplePerBond(reactant, rxns)
            rsbs = rmb.group_by_bond()
            tot = len(rsbs)
            bond_has_rxn = [True if len(x.reactions) > 0 else False for x in rsbs]
            num_bonds.append(tot)
            frac.append(sum(bond_has_rxn) / tot)

        print("### number of bonds in dataset (mean):", np.mean(num_bonds))
        print("### number of bonds in dataset (median):", np.median(num_bonds))
        print("### broken bond ratio in dataset (mean):", np.mean(frac))
        print("### broken bond ratio in dataset (median):", np.median(frac))

    def write_bond_energies(self, filename):
        """
        Write bond energy to a yaml file.
        """

        groups = self.group_by_reactant_all()

        # convert to nested dict: new_groups[reactant_idx][bond][charge] = rxn
        new_groups = OrderedDict()
        for rmb in groups:
            m = rmb.reactant
            key = "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
            new_groups[key] = OrderedDict()
            rsbs = rmb.group_by_bond()
            for rsb in rsbs:
                bond = rsb.broken_bond
                new_groups[key][bond] = OrderedDict()
                for rxn in rsb.reactions:
                    charge = tuple([m.charge for m in rxn.products])
                    new_groups[key][bond][charge] = rxn.as_dict()

        yaml_dump(new_groups, filename)

    def create_struct_label_dataset_reaction_based_regression_general(
        self,
        group_mode="all",
        one_per_iso_bond_group=True,
        sdf_mapping=False,
        extra_info=False,
    ):
        """
        Write the reaction

        This is based on reaction:

        Each reaction uses molecule instances for its reactants and products. As a
        result, a molecule is represented multiple times, which takes long time.

        Args:
            #struct_file (str): filename of the sdf structure file
            #label_file (str): filename of the label
            #feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.
            sdf_mapp(bool): use sdf mapping or just bond index
        """

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        all_mols, all_mol_ids, all_labels = [], [], []
        print("number of grouped reactions: {}".format(len(grouped_rxns)))

        #for grp in grouped_rxns:
        print("---> generating grouped reactions")
        for grp in tqdm(grouped_rxns, desc="grouped reactions"):
            reactions = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions=False
            )
            # rxn: a reaction for one bond and a specific combination of charges
            for i, rxn in enumerate(reactions):
                mols = rxn.reactants + rxn.products
                mols_id = [
                    str(i.id) + "_" + str(ind) for ind, i in enumerate(rxn.reactants)
                ] + [str(i.id) + "_" + str(ind) for ind, i in enumerate(rxn.products)]
                all_mols.extend(mols)
                all_mol_ids.extend(mols_id)
        print("--> generating labels")
        
        for grp in tqdm(grouped_rxns, desc="labeled reactions"):
        
        #for grp in grouped_rxns:
        
        # tqdm(grouped_rxns, desc="grouped reactions"):
        
            reactions = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions=False
            )

            # rxn: a reaction for one bond and a specific combination of charges
            for i, rxn in enumerate(reactions):
                reactant_id = [
                    str(i.id) + "_" + str(ind) for ind, i in enumerate(rxn.reactants)
                ]
                product_ids = [
                    str(i.id) + "_" + str(ind) for ind, i in enumerate(rxn.products)
                ]
                energy = rxn.get_free_energy()
                rev_value = rxn.get_rev_energy()

                if sdf_mapping:
                    mapping = rxn.bond_mapping_by_sdf_int_index()
                else:
                    mapping = rxn.bond_mapping_by_int_index()
                data = {
                    "value": [energy],
                    "value_rev": [rev_value],
                    "num_mols": len(reactant_id + product_ids),
                    "products": [all_mol_ids.index(prod_id) for prod_id in product_ids],
                    "reactants": [
                        all_mol_ids.index(react_id) for react_id in reactant_id
                    ],
                    "atom_mapping": rxn.atom_mapping(),
                    "bond_mapping": mapping,
                    "total_bonds": rxn.total_bonds,
                    "total_atoms": rxn.total_atoms,
                    "num_atoms_total": rxn.num_atoms_total,
                    "num_bonds_total": rxn.num_bonds_total,
                    "broken_bonds": rxn.get_broken_bond(),
                    "formed_bonds": rxn.get_formed_bond(),
                    "id": [rxn.get_id()],
                    "reaction_type": rxn.get_type(),
                }
                if extra_info:
                    data["extra_info"] = rxn.get_extra_info()
                all_labels.append(data)

        # write sdf - should be the same but we might need to get indeces again
        # self.write_sdf_custom(all_mols, struct_file)
        # label file
        # yaml_dump(all_labels, label_file)
        # write feature - done
        # if feature_file is not None:
        #    features = self.get_feature(all_mols, bond_indices=None)
        #    # extra features (global) that could be included
        #    features = [{"charge": i["charge"]} for i in features]
        #    yaml_dump(features, feature_file)

        features = self.get_feature(all_mols, bond_indices=None)
        features = [{"charge": i["charge"]} for i in features]
        print("features: {}".format(len(features)))
        print("labels: {}".format(len(all_labels)))
        print("molecules: {}".format(len(all_mols)))

        return all_mols, all_labels, features

    def create_struct_label_dataset_mol_based(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        lowest_across_product_charge=True,
    ):
        """
        Write the reactions to files.

        The is molecule based, each molecule will have a line in the label file.

        args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            lowest_across_product_charge (bool): If `True` each reactant corresponds to
                the lowest energy products. If `False`, find all 0->0+0 reactions,
                i.e. the charge of reactant and products should all be zero.

        """
        if lowest_across_product_charge:
            grouped_reactions = self.group_by_reactant_lowest_energy()
        else:
            grouped_reactions = self.group_by_reactant_charge_0()

        # write label
        create_directory(label_file)
        with open(to_path(label_file), "w") as f:
            f.write(
                "# Each line lists the bond energies of a molecule. "
                "The number of items in each line is equal to 2*N, where N is the "
                "number bonds. The first N items are bond energies and the next N "
                "items are indicators (0 or 1) to specify whether the bond energy "
                "exists in the dataset. A value of 0 means the corresponding bond "
                "energy should be ignored, whatever its value is.\n"
            )

            for rsr in grouped_reactions:
                reactant = rsr.reactant

                # get a mapping between sdf bond and reactions
                rxns_by_sdf_bond = dict()
                for rxn in rsr.reactions:
                    bond = rxn.get_broken_bond()
                    rxns_by_sdf_bond[bond] = rxn

                # write bond energies in the same order as sdf file
                energy = []
                indicator = []
                sdf_bonds = reactant.get_sdf_bond_indices(zero_based=True)
                for ib, bond in enumerate(sdf_bonds):
                    # have reaction that breaks this bond
                    if bond in rxns_by_sdf_bond:
                        rxn = rxns_by_sdf_bond[bond]
                        energy.append(rxn.get_free_energy())
                        indicator.append(1)
                    else:
                        energy.append(0.0)
                        indicator.append(0)

                for i in energy:
                    f.write("{:.15g} ".format(i))
                f.write("    ")
                for i in indicator:
                    f.write("{} ".format(i))
                f.write("\n")

        # write sdf
        reactants = [rsr.reactant for rsr in grouped_reactions]
        self.write_sdf(reactants, struct_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(reactants, bond_indices=None)
            yaml_dump(features, feature_file)

    @staticmethod
    def write_sdf(molecules, filename="struct.sdf"):
        """
        Write molecules to sdf.

        Args:
            molecules (list): a sequence of :class:`MoleculeWrapper`
            filename (str): output filename
        """
        logger.info("Start writing sdf file: {}".format(filename))

        create_directory(filename)
        with open(to_path(filename), "w") as f:
            for i, m in enumerate(molecules):
                # else: charge = m.charge
                name = "{}_{}_{}_{}_index-{}".format(
                    m.id, m.formula, m.charge, m.free_energy, i
                )
                sdf = m.write(name=name)
                f.write(sdf)

        logger.info("Finish writing sdf file: {}".format(filename))

    @staticmethod
    def write_sdf_custom(molecules, filename="struct.sdf"):
        """
        Write molecules to sdf.

        Args:
            molecules (list): a sequence of :class:`MoleculeWrapper`
            filename (str): output filename
        """
        logger.info("Start writing sdf file: {}".format(filename))

        create_directory(filename)
        with open(to_path(filename), "w") as f:
            for i, m in enumerate(molecules):
                sdf = m.write_custom(index=i)
                f.write(sdf)

        logger.info("Finish writing sdf file: {}".format(filename))

    @staticmethod
    def get_feature(molecules, bond_indices=None):
        """
        Get features from molecule.

        Args:
            molecules (list): a sequence of :class:`MoleculeWrapper`
            bond_indices (list of tuple or None): broken bond in the corresponding
                molecule

        Returns:
            list: features extracted from wrapper molecule, a dict for each molecule.
        """

        all_feats = []
        for i, m in enumerate(molecules):
            if bond_indices is None:
                idx = None
            else:
                idx = bond_indices[i]
            feat = m.pack_features(broken_bond=idx)
            if "index" not in feat:
                feat["index"] = i
            all_feats.append(feat)

        return all_feats

    def create_input_files(
        self,
        mol_file="molecules.sdf",
        mol_attr_file="attr.yaml",
        rxn_file="rxn.yaml",
    ):
        """
        Convert the reaction to input files expected from an end user.

        This is similar to `create_regression_dataset_reaction_network_simple` but
        write out different files.

        Args:
            mol_file (str): path to the sdf structure file of molecules
            mol_attr_file (str): path to the attributes of molecules (e.g. charge)
            rxn_file (str): path to the file of reactions

        """

        # all molecules in existing reactions
        reactions = self.reactions
        mol_reservoir = get_molecules_from_reactions(reactions)
        mol_reservoir = sorted(mol_reservoir, key=lambda m: m.id)
        mol_id_to_index_mapping = {m.id: i for i, m in enumerate(mol_reservoir)}

        rxns = []
        for i, rxn in enumerate(reactions):
            # change to index (in mol_reservoir) representation
            reactant_ids = [mol_id_to_index_mapping[m.id] for m in rxn.reactants]
            product_ids = [mol_id_to_index_mapping[m.id] for m in rxn.products]

            rxns.append(
                {
                    "index": i,
                    "reactants": reactant_ids,
                    "products": product_ids,
                    "energy": rxn.get_free_energy(),
                }
            )

        # mol file
        self.write_sdf(mol_reservoir, mol_file)

        # attr file
        attr = []
        for i, m in enumerate(mol_reservoir):
            attr.append({"charge": m.charge})
        yaml_dump(attr, mol_attr_file)

        # reaction file
        yaml_dump(rxns, rxn_file)


def get_molecules_from_reactions(reactions):
    """Return a list of unique molecules participating in all reactions."""
    mols = set()
    # mols = []
    for r in reactions:
        mols.update(r.reactants + r.products)
    return list(mols)


def get_atom_bond_mapping(rxn):
    atom_mp = rxn.atom_mapping()
    bond_mp = rxn.bond_mapping_by_sdf_int_index()
    return atom_mp, bond_mp


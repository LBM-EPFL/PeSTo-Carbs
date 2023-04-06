import os
import torch as pt
from tqdm import tqdm
from glob import glob

from src.dataset import StructuresDataset, collate_batch_features
from src.data_encoding import encode_structure, encode_features, extract_topology
from src.structure import encode_bfactor, concatenate_chains, split_by_chain
from src.structure_io import save_pdb

# load functions
from config import config_model
from model import Model


def apply_model(data_path):
    # define device
    device = pt.device("cpu")

    # create model
    model = Model(config_model)

    # reload model
    model.load_state_dict(pt.load(os.path.join("./", 'model_ckpt.pt'), map_location=pt.device("cpu")))

    # set model to inference
    model = model.eval().to(device)

    # find pdb files and ignore already predicted oins
    pdb_filepaths = glob(os.path.join(data_path, "*.pdb1"), recursive=True)
    pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]

    # create dataset loader with preprocessing
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    # debug print
    print("Test size: {0} ".format(len(dataset)))

    # run model on all subunits
    with pt.no_grad():
        for subunits, filepath in tqdm(dataset):
            print(filepath)
            # concatenate all chains together
            structure = concatenate_chains(subunits)

            # encode structure and features
            X, M = encode_structure(structure)
            #q = pt.cat(encode_features(structure), dim=1)
            q = encode_features(structure)[0]

            # extract topology
            ids_topk, _, _, _, _ = extract_topology(X, 64)

            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # run model
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

            # for all predictions
            for i in range(z.shape[1]):
                # prediction
                p = pt.sigmoid(z[:,i])

                # encode result
                structure = encode_bfactor(structure, p.cpu().numpy())

                # save results
                output_filepath = filepath[:-5]+'_i{}.pdb'.format(i)
                save_pdb(split_by_chain(structure), output_filepath)


if __name__ == '__main__':
    apply_model("pdbs")

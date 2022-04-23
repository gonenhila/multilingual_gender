
import torch
from transformers import BertModel, BertForMaskedLM, BertTokenizer, BertConfig
import transformers
import numpy as np
import pdb
import argparse
from tqdm import tqdm

def forward_from_specific_layer(model: transformers.BertModel, layer_number: int,
                                layer_representation: torch.Tensor):
    """
   :param model: a BertForMaskedLM model
   :param layer_representation: a torch tensor, dims: [1, seq length, 768]
   Return:
           states, a numpy array. dims: [#LAYERS - layer_number, seq length, 768]
           last_state_after_batch_norm: np array, after batch norm. dims: [seq_length, 768]
   """

    layers = model.bert.encoder.layer[layer_number:]
    layers.append(model.cls.predictions.transform)

    h = layer_representation
    states = []

    with torch.no_grad():
        for i, layer in enumerate(layers):
            h = layer(h)[0] if i != len(layers) - 1 else layer(h)
            states.append(h)

    for i, s in enumerate(states):
        states[i] = s.detach().cpu().numpy()

    states = np.array(states)
    for x in states:
        assert len(x.shape) == 3

    return states.squeeze(1)


def intervene_in_layer(model: transformers.BertModel, representations: torch.Tensor, layer_number: int,
                       projection_matrix: torch.tensor, apply_on_all=True):
    """
    Intervening in the representations at layer layer_number.
    :param model: a bert model
    :layer_number: the layer on which we intervene.
    :param projection_matrix
    :param apply_on_all: if True, apply projection on all tokens. O/w, apply on the CLS only.
    Returns: the all hidden representations of the model.
    """

    # extract representation at layer i

    # with torch.no_grad():
    #     outputs = model(representations, return_dict=True)
    # pdb.set_trace()
    # hidden_states = outputs["hidden_states"]
    # hidden_state_layer_i = hidden_states[layer_number][0] # (seq_len, 768)
    # if apply_on_all:
    #
    #     hidden_state_layer_i = hidden_state_layer_i @ projection_matrix
    # else:
    #     hidden_state_layer_i[0] = hidden_state_layer_i[0] @ projection_matrix

    batches = [representations[i:i+64,:] for i in range(0, representations.shape[0], 64)]
    all_final = []

    for batch in tqdm(batches):
        # continue the forward pass
        batch = torch.from_numpy(batch)
        hidden_state_layer_i = batch.unsqueeze(0)  # add empty batch dim
        hidden_after_projection_i_onwards = forward_from_specific_layer(model, layer_number, hidden_state_layer_i)
        #hidden_states_until_i = torch.stack(hidden_states[:layer_number]).squeeze(1)
        #hidden_states_after_projection = torch.cat([hidden_states_until_i, torch.tensor(hidden_after_projection_i_onwards)])
        #return hidden_states_after_projection
        all_final.append(hidden_after_projection_i_onwards[-1])
        #print("hidden_after_projection_i_onwards[-1]", hidden_after_projection_i_onwards[-1])
        #print("len", len(hidden_after_projection_i_onwards[-1]))
    all_final = np.concatenate(all_final, axis=0)
    return all_final


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="EN", help="language")
    parser.add_argument("--layer", type=int, help="layer to use")
    parser.add_argument("--type", help="cls or avg")
    parser.add_argument("--inlp_mat", help="inlp matrix to load")

    args = parser.parse_args()

    config = BertConfig.from_pretrained("bert-base-multilingual-uncased", output_hidden_states=True)
    bert = BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased", config=config)


    path = "../data/bert_encode_biasbios/{}/".format(args.lang)
    x_train = np.load(path + "train_{}_layer{}_mbert.npy".format(args.type, args.layer))
    x_dev = np.load(path + "dev_{}_layer{}_mbert.npy".format(args.type, args.layer))
    x_test = np.load(path + "test_{}_layer{}_mbert.npy".format(args.type, args.layer))
    P = np.load(args.inlp_mat)

    x_train_after = intervene_in_layer(bert, x_train, args.layer, P, apply_on_all=False)
    x_dev_after = intervene_in_layer(bert, x_dev, args.layer, P, apply_on_all=False)
    x_test_after = intervene_in_layer(bert, x_test, args.layer, P, apply_on_all=False)

    with open(path + "train_{}_layer{}_mbert_final_not_all.npy".format(args.type, args.layer), 'wb') as f:
        np.save(f, x_train_after)
    with open(path + "dev_{}_layer{}_mbert_final_not_all.npy".format(args.type, args.layer), 'wb') as f:
        np.save(f, x_dev_after)
    with open(path + "test_{}_layer{}_mbert_final_not_all.npy".format(args.type, args.layer), 'wb') as f:
        np.save(f, x_test_after)



if __name__ == '__main__':
    main()



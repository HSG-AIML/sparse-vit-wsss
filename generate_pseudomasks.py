import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import cv2
from tqdm.autonotebook import tqdm

import datasets
from models.vit import ViT, MViT
from models.recorder import Recorder


TARGET_TASK_MAP = {
    "dfc": "multi-label",
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def parse_args():
    parser = argparse.ArgumentParser()
    #  Parameters
    parser.add_argument("--depth", default=12, type=int, help="Total number of blocks.")
    parser.add_argument("--patch_size", default=12, type=int, help="Patch Size.")
    parser.add_argument("--num_classes", default=8, type=int, help="Number of classes")
    parser.add_argument("--num_heads", default=16, type=int, help="Number of heads")
    parser.add_argument(
        "--num_channels", default=13, type=int, help="Number of channels"
    )
    parser.add_argument("--dataset", default="dfc", type=str, help="Dataset Selected")
    parser.add_argument("--pr_rate", default=0.2, type=float, help="Pruning coef")
    parser.add_argument(
        "--mult", default=1, type=int, help="Multiplication coefficient"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Total batch size.")
    parser.add_argument(
        "--arch",
        default="vit",
        type=str,
        help="Architecture desired - Vit, DeepViT etc.",
    )
    parser.add_argument(
        "--exp_name", default="900_5000", type=str, help="Name of experiment"
    )
    parser.add_argument(
        "--datadir", default="/path/to/data", type=str, help="Path to data directory"
    )
    parser.add_argument(
        "--model_checkpoint",
        default="/path/to/model/checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument("--imgsize", nargs="+", type=int)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    return parser.parse_args()


def to_uint8(img):
    return (img * 255).astype("uint8")


def fuse_class_maps(images, classes=False):
    if classes:
        fused = np.ones(images[0].shape) * -1
    else:
        fused = np.zeros(images[0].shape)
    for idx, img in enumerate(images):
        if not classes:
            img = (idx + 1) * img
        fused[img > -1] = img[img > -1]

    return fused


def get_heads(att, patchsize=8, layer=11, imgsize=224):
    heads = []
    indices = []
    for i in range(len(att[layer])):
        if sum(sum(att[layer][i])) != 0:
            indices.append(i)

    s0 = int(imgsize[0] / patchsize)
    s1 = int(imgsize[1] / patchsize)
    for i in range(len(indices)):
        try:
            mask = att[layer, indices[i], :, :][0, 1:].reshape(s1, s0).detach().numpy()
            mask_r = cv2.resize(mask / mask.max(), imgsize)[..., np.newaxis]
            heads.append(mask_r)
        except IndexError:
            pass

    heads = np.array(heads)
    heads = np.reshape(heads, (len(heads), -1))
    return heads


def cluster_heads(n_clusters, heads, size=16, seed=22):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(heads)

    mean_classes = []
    for c in range(n_clusters):
        class_c = []
        for idx, lbl in enumerate(kmeans.labels_):
            if lbl == c:
                class_c.append(np.reshape(heads[idx], size))

        class_c = np.array(class_c)
        mean_class_c = np.mean(class_c, 0)

        mean_classes.append(mean_class_c)

    return np.array(mean_classes)


def cluster_means_perpixel(n_clusters, means, seed=21):
    predictions = []
    mean_values = []
    for i in range(len(means)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(means[i].flatten().reshape(-1, 1))
        preds = kmeans.predict(means[i].flatten().reshape(-1, 1))

        preds_tmp = np.zeros(preds.shape)

        mean_values.append(
            max(
                np.mean(means[i].flatten()[preds == 0]),
                np.mean(means[i].flatten()[preds == 1]),
            )
        )

        if np.mean(means[i].flatten()[preds == 0]) > np.mean(
            means[i].flatten()[preds == 1]
        ):
            preds_tmp[preds.copy() == 0] = 1
        else:
            preds_tmp[preds.copy() == 1] = 1
        predictions.append(preds_tmp)
    return np.array(predictions), mean_values


def nearest_valid_entry_2d(a, x, y):
    idx = np.argwhere(a != -1)
    return a[
        idx[((idx - [x, y]) ** 2).sum(1).argmin()][0],
        idx[((idx - [x, y]) ** 2).sum(1).argmin()][1],
    ]


def generate_pseudomask(model, mean_classes, size, predicted, probs, data, oracle):
    probs = probs.detach().numpy()
    predictions, mean_preds = cluster_means_perpixel(2, mean_classes)
    if oracle:
        rounded_probs = predicted
    else:
        rounded_probs = np.round(probs)[0]
    classes = np.where(rounded_probs == 1)[0]
    ordered_classes = classes[np.argsort(probs.squeeze()[classes])[::-1]]

    for i in range(len(np.argsort(mean_preds)[::-1])):
        value = ordered_classes[i]
        predictions[np.argsort(mean_preds)[::-1][i]] = np.where(
            predictions[np.argsort(mean_preds)[::-1][i]] == 0,
            -1,
            predictions[np.argsort(mean_preds)[::-1][i]],
        )
        predictions[np.argsort(mean_preds)[::-1][i]] = np.where(
            predictions[np.argsort(mean_preds)[::-1][i]] == 1,
            predictions[np.argsort(mean_preds)[::-1][i]] * value,
            -1,
        )
    fused_predictions = fuse_class_maps(
        predictions[np.argsort(mean_preds)], True
    ).reshape(size)

    fused_predictions_valid = fused_predictions.copy()
    for i in range(0, len(fused_predictions_valid)):
        for j in range(0, len(fused_predictions_valid)):
            if fused_predictions[i, j] == -1:
                fused_predictions_valid[i, j] = nearest_valid_entry_2d(
                    fused_predictions, i, j
                )
    new_fused_predictions_valid = forward_pass(model, data, fused_predictions_valid)
    return fused_predictions_valid, new_fused_predictions_valid


def forward_pass(model, data, pseudomask):
    model.eval()
    new_pseudomask = pseudomask.copy()
    for label in np.unique(pseudomask):
        sigmoid = torch.nn.Sigmoid()
        temp_data = data.clone()
        temp_data[:, :, pseudomask != label] = 0
        with torch.no_grad():
            out, _ = model(temp_data)
            out = sigmoid(out)
            predicted = np.argmax(out)
            new_pseudomask[pseudomask == label] = predicted
    return new_pseudomask


def setup_model(
    arch,
    imgsize,
    num_classes,
    depth,
    patch_size,
    num_heads,
    prune,
    num_channels,
    pr_rate,
    multimodal,
    model_checkpoint,
):
    if multimodal:
        model = MViT(
            image_size=tuple(imgsize),
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=depth,
            heads=num_heads,
            mlp_dim=2048,
            device=device,
            m1_channels=2,
            m2_channels=13,
            prune=prune,
            dropout=0.0,
            emb_dropout=0.0,
            l0_penalty=pr_rate,
        )
    else:
        model = ViT(
            image_size=tuple(imgsize),
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=depth,
            heads=num_heads,
            mlp_dim=2048,
            device=device,
            prune=prune,
            channels=num_channels,
            dropout=0.0,
            emb_dropout=0.0,
            l0_penalty=pr_rate,
        )

    model_weights = torch.load(model_checkpoint, map_location="cpu")["state_dict"]
    new_dict = model_weights.copy()

    for key, _ in model_weights.items():
        new_key = key.replace("model.", "")
        new_dict[new_key] = new_dict.pop(key)

    model.load_state_dict(new_dict)

    return Recorder(model, prune)


def save_pseudomaks(
    pseudo_masks, pseudo_masks_old, real_masks, valid_indices, exp_name
):
    pseudomask_path = "pseudomasks"
    if not os.path.exists(pseudomask_path):
        os.makedirs(pseudomask_path)

    np.save(
        os.path.join(pseudomask_path, "pseudo_masks_{}".format(exp_name)), pseudo_masks
    )
    np.save(
        os.path.join(pseudomask_path, "pseudo_masks_old_{}".format(exp_name)),
        pseudo_masks_old,
    )
    np.save(os.path.join(pseudomask_path, "real_masks_{}".format(exp_name)), real_masks)
    np.save(
        os.path.join(pseudomask_path, "valid_indices_{}".format(exp_name)),
        valid_indices,
    )


def create_pseudo_groundtruh(
    model, train_loader, multimodal, oracle, exp_name, patchsize=14, imgsize=224
):
    model.eval()

    valid_indices = []
    pseudo_masks = []
    pseudo_masks_old = []
    real_masks = []

    progress = tqdm(
        enumerate(train_loader), desc="Train Loss: ", total=len(train_loader)
    )

    for idx, batch in progress:
        with torch.no_grad():
            if multimodal:
                s1 = batch["s1"].float().to(device)
                s2 = batch["img"].float().to(device)
                t_mask = batch["mask"]
                output, att_mat = model(s1, s2)
            else:
                data = batch["img"].float().to(device)
                t_mask = batch["mask"]
                output, att_mat = model(data)

            predicted = np.round(output.detach().numpy())[0]
            n_label = np.sum(predicted)

        if n_label == 1:
            valid_indices.append(idx)
            mask_cluster = np.ones(imgsize) * np.argmax(output.detach().numpy())
            pseudo_masks.append(mask_cluster)
            pseudo_masks_old.append(mask_cluster)
            real_masks.append(t_mask)
        if n_label > 1:
            try:
                valid_indices.append(idx)
                heads = get_heads(
                    att_mat[0], patchsize=patchsize, layer=-1, imgsize=imgsize
                )

                mean_classes = cluster_heads(int(n_label), heads, size=imgsize, seed=24)
                mask_cluster_old, mask_cluster = generate_pseudomask(
                    model, mean_classes, imgsize, predicted, output, data, oracle
                )

                pseudo_masks.append(mask_cluster)
                pseudo_masks_old.append(mask_cluster_old)
                real_masks.append(t_mask)
            except (ValueError, UnboundLocalError):
                pass
        print("\r" + "sample nb {}".format(idx), end="", flush=True)

        if idx % 10 == 0 and idx != 0:
            save_pseudomaks(
                pseudo_masks, pseudo_masks_old, real_masks, valid_indices, exp_name
            )

    return pseudo_masks, pseudo_masks_old, real_masks, valid_indices


def create_dataset(dataset, datadir, mult, train_batch_size, eval_batch_size):
    if dataset == "dfc":
        train_loader, test_loader = datasets.dfc.create(
            datadir, mult, train_batch_size, eval_batch_size
        )
    else:
        print("The dataset doesn't exist.")
        return

    return train_loader, test_loader


def main():
    args = parse_args()

    model = setup_model(
        args.arch,
        args.imgsize,
        args.num_classes,
        args.depth,
        args.patch_size,
        args.num_heads,
        args.prune,
        args.num_channels,
        args.pr_rate,
        args.multimodal,
        args.model_checkpoint,
    )

    _, train_loader = create_dataset(
        args.dataset,
        args.datadir,
        args.mult,
        args.imgsize,
        args.batch_size,
        args.batch_size,
    )

    (
        pseudo_masks,
        pseudo_masks_old,
        real_masks,
        valid_indices,
    ) = create_pseudo_groundtruh(
        model,
        train_loader,
        args.multimodal,
        args.oracle,
        args.exp_name,
        args.patch_size,
        tuple(args.imgsize),
    )

    save_pseudomaks(
        pseudo_masks, pseudo_masks_old, real_masks, valid_indices, args.exp_name
    )


if __name__ == "__main__":
    main()

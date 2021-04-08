import os

import numpy as np
import torch
from tqdm import tqdm

from definitions import Keyframe, path_data
from training_config import Train_Config


def __frame_to_feature(frame, featureset, config):
    return featureset.extract(Keyframe.from_numpy(frame))


def __frames_to_feature_batch(frames, featureset, config):
    return featureset.extract_batch(frames, random_occlusion=config.random_occlusion)


def to_recurrent_feature(X, y, sequence_length, sequence_distance, sequence_skip):
    _X = torch.empty(((X.shape[0] - (sequence_length * sequence_distance)) // sequence_skip + 1, sequence_length, X.shape[1]))

    for i in range(sequence_length):
        end_idx = -(sequence_length * sequence_distance - (i * sequence_distance) - 1)
        if end_idx == 0:
            end_idx = X.shape[0]
        _X[:, i] = X[i * sequence_distance:end_idx:sequence_skip]

    if sequence_distance == 1:
        return _X, y[(sequence_length - 1) * sequence_distance::sequence_skip]
    else:
        return _X, y[(sequence_length - 1) * sequence_distance:-(sequence_distance - 1):sequence_skip]


def to_recurrent_feature_index(X, y, sequence_length, sequence_skip, i0, i1=0, id=1):
    i1 = i1 if i1 != 0 else len(X)
    x0 = torch.arange(i0, i1, id)
    _X = torch.empty((len(x0), sequence_length, X.shape[1]))

    for s in range(sequence_length):
        s0 = x0 - (sequence_length - 1 - s) * sequence_skip
        s0[s0 < 0] = 0
        _X[:, s] = X[s0]
    return _X, y[i0:i1:id]


def create_training_data_config(parser, c: Train_Config, update, save_all_to_mem=True, shuffle_all=True, dataset_name="", test_set=False):
    in_features = c.in_features.data_to_features()
    out_features = c.out_features.data_to_features()

    if dataset_name == "":
        raise Exception("ERROR: no dataset_name set")

    if shuffle_all:
        save_all_to_mem = True

    _suffix = "_" + dataset_name + ("_normalized" if c.normalized_bones else "")
    if update:
        data = parser.load_numpy(c.normalized_bones)
        batch_nr = 0
        suffix = f"{_suffix}{batch_nr}"
        while os.path.exists(os.path.join(path_data, in_features.name + suffix + ".dat")) or \
                os.path.exists(os.path.join(path_data, out_features.name + suffix + ".dat")):
            if os.path.exists(os.path.join(path_data, in_features.name + suffix + ".dat")):
                os.remove(os.path.join(path_data, in_features.name + suffix + ".dat"))
            if os.path.exists(os.path.join(path_data, out_features.name + suffix + ".dat")):
                os.remove(os.path.join(path_data, out_features.name + suffix + ".dat"))
            batch_nr += 1
            suffix = f"{_suffix}{batch_nr}"

        batch_nr = 0
        X = []
        y = []

        for skeleton, tracks in tqdm(data, desc="extracting features"):
            for _track in tracks:
                track = _track[0] if len(_track.shape) == 4 else _track
                _x = __frames_to_feature_batch(track, in_features, config=c)
                _y = __frames_to_feature_batch(track, out_features, config=c)

                X.extend(_x.reshape((-1, _x.shape[1])))
                y.extend(_y.reshape((-1, _y.shape[1])))

            while len(X) >= c.input_size:
                next_X = X[c.input_size:]
                next_y = y[c.input_size:]
                X = X[:c.input_size]
                y = y[:c.input_size]

                X = torch.from_numpy(np.array(X))
                if c.out_features is in_features:
                    y = X
                else:
                    y = torch.from_numpy(np.array(y))

                suffix = f"{_suffix}{batch_nr}"

                torch.save(X, os.path.join(path_data, in_features.name + suffix + ".dat"))
                torch.save(y, os.path.join(path_data, out_features.name + suffix + ".dat"))
                batch_nr += 1

                X = next_X
                y = next_y
        if len(X) > 0:
            X = torch.from_numpy(np.array(X))
            if c.out_features is in_features:
                y = X
            else:
                y = torch.from_numpy(np.array(y))

            suffix = f"{_suffix}{batch_nr}"

            torch.save(X, os.path.join(path_data, in_features.name + suffix + ".dat"))
            torch.save(y, os.path.join(path_data, out_features.name + suffix + ".dat"))

    suffix = f"{_suffix}0"
    # if data is not available, create it!
    if not os.path.exists(os.path.join(path_data, in_features.name + suffix + ".dat")) or \
            not os.path.exists(os.path.join(path_data, out_features.name + suffix + ".dat")):
        yield from create_training_data_config(parser, c, True, save_all_to_mem, shuffle_all, dataset_name)
        return

    batch_nr = 0
    batch_samples = 0
    X_batch = None
    y_batch = None
    suffix = f"{_suffix}{batch_nr}"
    X_all = None
    y_all = None
    while os.path.exists(os.path.join(path_data, in_features.name + suffix + ".dat")) and os.path.exists(
            os.path.join(path_data, out_features.name + suffix + ".dat")):
        X = torch.load(os.path.join(path_data, in_features.name + suffix + ".dat"))
        y = torch.load(os.path.join(path_data, out_features.name + suffix + ".dat"))
        input_size = c.input_size if c.input_size > 0 else len(X)
        assert (len(X.shape) == 2)

        if save_all_to_mem:
            if X_all is None or y_all is None:
                X_all = [X]
                y_all = [y]
            else:
                X_all.append(X)
                y_all.append(y)
        else:
            if X_batch is None:
                X_batch = torch.empty((input_size, X.shape[1]), dtype=torch.float32)
                y_batch = torch.empty((input_size, y.shape[1]), dtype=torch.float32)

            for batch in range(max(1, len(X) // input_size)):
                new_samples = min(len(X), input_size - batch_samples)
                X_batch[batch_samples:batch_samples + new_samples] = X[batch * input_size: batch * input_size + new_samples]
                y_batch[batch_samples:batch_samples + new_samples] = y[batch * input_size: batch * input_size + new_samples]

                if batch_samples + new_samples == input_size:
                    if c.model.recurrent:
                        yield to_recurrent_feature(X_batch, y_batch, c.sequence_length, c.sequence_distance,
                                                   c.sequence_skip)
                    else:
                        yield X_batch, y_batch
                    if batch == max(1, len(X) // input_size) - 1:
                        batch_samples = min(input_size, len(X) - new_samples)
                        if batch_samples > 0:
                            X_batch[:batch_samples] = X[-batch_samples:]
                            y_batch[:batch_samples] = y[-batch_samples:]
                    else:
                        batch_samples = 0
                else:
                    batch_samples += new_samples

        batch_nr += 1
        suffix = f"{_suffix}{batch_nr}"

    if save_all_to_mem and X_all is not None:
        X_all = torch.cat(X_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        if shuffle_all:
            batches = len(X_all) // input_size
            last_entry = len(X_all) - (len(X_all) % input_size)
            if c.model.recurrent:
                for i in range(0, batches, c.sequence_distance):
                    yield to_recurrent_feature_index(X_all, y_all, c.sequence_length, c.sequence_skip, i, last_entry, batches)
            else:
                for i in range(batches):
                    yield X_all[i:last_entry:batches], y_all[i:last_entry:batches]
        else:
            if test_set:
                X_all = X_all[X_all.shape[0] * 8 // 10:]
                y_all = y_all[y_all.shape[0] * 8 // 10:]
            if c.model.recurrent:
                yield to_recurrent_feature(X_all, y_all, c.sequence_length, c.sequence_distance, c.sequence_skip)
            else:
                yield X_all, y_all

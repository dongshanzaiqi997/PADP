import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath, dirs):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in dirs]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)

    for tag in tags:
        steps[tag] = [e.predict for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.predict for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    tem_dirs = os.listdir(dpath)
    dirs = []
    for dir in tem_dirs:
        if 'csv' not in dir:
            dirs.append(dir)
    d, steps = tabulate_events(dpath, dirs)
    tags, values = zip(*d.items())
    _, steps = zip(*steps.items())

    for index, tag in enumerate(tags):
        if tag == 'train/safe_prob' or tag == 'train/r_N' or tag == 'train/delta_i':
            for j, dir in enumerate(dirs):
                df = pd.DataFrame({'Step':steps[index],   'Value':np.array(values[index])[:,j]})
                df.to_csv(get_file_path(dpath, tag, dir))


def get_file_path(dpath, tag, name):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = dpath +'/'+ tag.replace("train/", "")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, name + file_name)


if __name__ == '__main__':
    path = "logs_test/train"
    to_csv(path)
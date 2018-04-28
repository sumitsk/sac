import datetime
import dateutil.tz
import os
import numpy as np
import pdb

def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')

def concat_obs_z(obs, z, num_skills, concat_type):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[z] = 1
    if concat_type == 'concatenation':
        return np.hstack([obs, z_one_hot])
    elif concat_type == 'bilinear':
        return np.outer(z_one_hot, obs).flatten()
    else:
        raise NotImplementedError


def split_aug_obs(aug_obs, num_skills, concat_type):
    """Splits an augmented observation into the observation and Z."""
    if concat_type == 'concatenation':
        (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
        z = np.where(z_one_hot == 1)[0][0]
        return (obs, z)
    elif concat_type == 'bilinear':
        splits = np.split(aug_obs, indices_or_sections=num_skills, axis=1)
        stack = np.stack(splits, axis=2)
        obs = np.sum(stack, axis=2)

        val = np.sum(np.abs(stack), axis=1)
        mask = np.not_equal(val, 0)
        z = mask * 1.0
        return obs, z

    else:
        raise NotImplementedError        


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def _save_video(paths, filename):
    import cv2
    assert all(['ims' in path for path in paths])
    ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def _softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))

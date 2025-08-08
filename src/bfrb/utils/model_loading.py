from collections import OrderedDict


def fix_model_state_dict(state_dict: dict):
    """
    :param state_dict:
    :return:
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.` prefix
        name = name.replace("_orig_mod.", "")

        name = name.replace(
            "positional_encoding", "positional_embedding"
        )  # seems this has been renamed
        new_state_dict[name] = v
    return new_state_dict
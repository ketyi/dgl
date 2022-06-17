"""We intend to make our reproduction as close as possible to the original paper.
The configuration in the file is mostly from the description in the original paper
and will be loaded when setting up."""


def dataset_based_configure(opts):

    if opts['dataset'] == 'cycles':
        ds_configure = cycles_configure
    else:
        raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    opts = {**opts, **ds_configure}

    return opts

# 'nepochs': 25,
synthetic_dataset_configure = {
    'node_hidden_size': 16,
    'num_propagation_rounds': 2,
    'optimizer': 'Adam',
    'nepochs': 1,
    'ds_size': 4000,
    'num_generated_samples': 10000,
}

cycles_configure = {
    **synthetic_dataset_configure,
    **{
        'min_size': 10,
        'max_size': 20,
        'lr': 5e-4,
        'node_types': ['event', 'application', 'resource', 'offer'],
        'edge_types': [':CORR', ':DF'],
        'correlation_attributes': ['application', 'resource', 'offer'],
        'edge_directions': ['to target', 'to source'],
        'activity_labels': ['create application', 'application ready', 'create offer', 'send offer', 'offer cancelled'],
        'canonical_etypes': [('event', ':DF', 'event'),
                             ('event', ':CORR', 'application'),
                             ('event', ':CORR', 'resource'),
                             ('event', ':CORR', 'offer')],
    }
}

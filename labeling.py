import gunpowder as gp
import json
import numpy as np
import os
import pathlib
import sys


class NetConfig:
    def __init__(self):
        with open(self.config_path(), 'r') as f:
            self.inner = json.load(f)

    def __getitem__(self, key):
        return self.inner[key]

    def config_path(self):
        raise NotImplementedError

    def graph_path(self):
        raise NotImplementedError

    def checkpoint_path(self, iteration):
        raise NotImplementedError


class FormatNetConfig(NetConfig):
    def __init__(self, label_name):
        self.label_name = label_name
        if self.label_name and not self.label_name.startswith('_'):
            self.label_name = '_' + self.label_name
        super().__init__()

    def config_path(self):
        return f'test_net{self.label_name}_config.json'

    def graph_path(self):
        return f'test_net{self.label_name}.meta'

    def checkpoint_path(self, iteration):
        return f'train_net{self.label_name}_checkpoint_{iteration}'
        raise NotImplementedError

class Predictor:
    def __init__(self, net_config, output_dataset):
        self.net_config = net_config
        self.output_dataset = output_dataset

    def predict(self, iteration, slab, n_slabs, dataset_file='dataset.json'):

        raw = gp.ArrayKey('RAW')
        pred_labels = gp.ArrayKey('PRED_LABELS')

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_dataset.json'), 'r') as f:
            DEFAULT_DATASET = json.load(f)
        if os.path.isfile(dataset_file):
            with open(dataset_file, 'r') as f:
                dataset = {**DEFAULT_DATASET, **json.load(f)}
        else:
            dataset = DEFAULT_DATASET

        print(json.dumps(dataset, sort_keys=True, indent=4))

        output_path = dataset['predict']['output']['container'] + self.output_dataset
        output_path_version = os.path.join(dataset['predict']['output']['container'], 'attributes.json')
        output_attr_path = os.path.join(output_path, 'attributes.json')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(output_path_version):
            with open(output_path_version, 'w') as f:
                json.dump({"n5": "2.0.2"}, f)
        if not os.path.isfile(output_attr_path):
            input_path = dataset['predict']['input']['container'] + dataset['predict']['input']['dataset']
            input_path = os.path.join(input_path, 'attributes.json')
            with open(input_path, 'r') as f:
                input_attr = json.load(f)
            output_attr = {
                "dataType": "int64",
                "compression": {"type": "gzip", "level": 5},
                "offset": [0, 0, 0],
                "resolution": dataset['predict']['input']['resolution'],
                "dimensions": input_attr['dimensions'],
                "blockSize": input_attr['blockSize'],
            }
            with open(output_attr_path, 'w') as f:
                json.dump(output_attr, f)

        SECTIONS=64

        # get the input and output size in world units (nm, in this case)
        voxel_size = gp.Coordinate(tuple(dataset['predict']['input']['resolution']))
        input_size = gp.Coordinate([SECTIONS,] + self.net_config['input_shape'])*voxel_size
        output_size = gp.Coordinate([SECTIONS,] + self.net_config['output_shape'][1:])*voxel_size
        context = input_size - output_size

        # formulate the request for what a batch should contain
        request = gp.BatchRequest()
        request.add(raw, input_size)
        request.add(pred_labels, output_size)

        #############################
        # ASSEMBLE TESTING PIPELINE #
        #############################
        slab_size = 1.0 / n_slabs
        slab_start = slab * slab_size
        slab_stop = (slab + 1) * slab_size

        source = gp.N5Source(
                dataset['predict']['input']['container'],
                {
                    raw: dataset['predict']['input']['dataset'],
                },
                {
                    raw: gp.ArraySpec(voxel_size=voxel_size, interpolatable=True),
                }
            ) + \
            gp.Pad(raw, context) + \
            gp.Crop(
                    raw,
                    fraction_negative=(slab_start, 0, 0),
                    fraction_positive=(1 - slab_stop, 0, 0))

        squeeze = gp.Squeeze(
                {
                    raw: 0,
                })

        # get the ROI provided for raw (we need it later to calculate the ROI in
        # which we can make predictions)
        with gp.build(source):
            raw_roi = source.spec[raw].roi

        pred_labels_roi = raw_roi.grow(-context, -context)
        pred_labels_2d_roi = gp.Roi(
                offset=pred_labels_roi.get_offset()[1:3],
                shape=pred_labels_roi.get_shape()[1:3])
        pred_labels_sq_roi = gp.Roi(
                offset=(pred_labels_roi.get_offset()[0],),
                shape=(pred_labels_roi.get_shape()[0],))

        expand = gp.Expand(
                axes={
                    pred_labels: {
                        'axis': 0,
                        'spec': gp.ArraySpec(roi=pred_labels_sq_roi, voxel_size=gp.Coordinate((voxel_size[0],))),
                        'propagate': False,
                    },
                },
                squeeze_node=squeeze
            )

        pipeline = (

            # read from HDF5 file
            source +

            # convert raw to float in [0, 1]
            gp.Normalize(raw) +

            #squeeze +

            # perform one training iteration for each passing batch (here we use
            # the tensor names earlier stored in train_net.config)
            gp.tensorflow.Predict(
                graph=self.net_config.graph_path(),
                checkpoint=self.net_config.checkpoint_path(iteration),
                inputs={
                    self.net_config['raw']: raw
                },
                outputs={
                    self.net_config['pred_labels']: pred_labels
                },
                array_specs={
                    pred_labels: gp.ArraySpec(roi=pred_labels_roi, dtype=np.uint32)
                }) +

            #expand.stubs() +

            #expand +

            # store all passing batches in the same HDF5 file
            gp.N5Write(
                {
                    #raw: '/volumes/raw',
                    pred_labels: self.output_dataset,
                },
                output_filename=dataset['predict']['output']['container'],
                compression_type='gzip'
            ) +

            # show a summary of time spend in each node every 10 iterations
            gp.PrintProfilingStats(every=10) +

            # iterate over the whole dataset in a scanning fashion, emitting
            # requests that match the size of the network
            gp.Scan(reference=request)
        )

        with gp.build(pipeline):
            # request an empty batch from Scan to trigger scanning of the dataset
            # without keeping the complete dataset in memory
            pipeline.request_batch(gp.BatchRequest())

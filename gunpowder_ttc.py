from __future__ import print_function
import gunpowder as gp
import json
import logging
import math
import os
import sys

def train(iterations, run_name="default"):

    ##################
    # DECLARE ARRAYS #
    ##################

    # raw intensities
    raw = gp.ArrayKey('RAW')

    # objects labelled with unique IDs
    gt_labels = gp.ArrayKey('LABELS')

    # weights to use to balance the loss
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')

    # the predicted labels
    pred_labels = gp.ArrayKey('PRED_LABELS')

    # the gredient of the loss wrt to the predicted labels
    pred_labels_gradients = gp.ArrayKey('PRED_LABELS_GRADIENTS')


    NUM_CLASSES = 3

    ####################
    # DECLARE REQUESTS #
    ####################

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)

    # get the input and output size in world units (nm, in this case)
    voxel_size = gp.Coordinate((48, 48, 48))
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size[1:]
    output_size = gp.Coordinate(net_config['output_shape'][1:])*voxel_size[1:]

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_labels, output_size)
    request.add(loss_weights, output_size)

    # when we make a snapshot for inspection (see below), we also want to
    # request the predicted affinities and gradients of the loss wrt the
    # affinities
    snapshot_request = gp.BatchRequest()
    snapshot_request[pred_labels] = request[gt_labels]
    snapshot_request[pred_labels_gradients] = request[gt_labels]

    expand = gp.Expand(
        axes={
            raw: {
                'axis': 0,
                'spec': gp.ArraySpec(roi=gp.Roi((0,),(48,)), voxel_size=gp.Coordinate((48,))),
                'propagate': False,
            },
            gt_labels: {
                'axis': 0,
                'spec': gp.ArraySpec(roi=gp.Roi((0,),(48,)), voxel_size=gp.Coordinate((48,))),
                'propagate': False,
            },
        })

    ##############################
    # ASSEMBLE TRAINING PIPELINE #
    ##############################

    pipeline = (

        (
            # Labels:
            # 0: background
            # 1: empty
            # 2: bundles
            # 3: neuropil
            # 4: tissue
            tuple(
                gp.DirectorySource(
                    'data/1018/0001 - VNC',
                    {

                        raw: 'z=0.0 to z=117000{i}-1.tif'.format(i=i),
                        gt_labels: 'label_render/ttc/labels000{i}.tif'.format(i=i)
                    },
                    {
                        raw: gp.ArraySpec(voxel_size=(48,48), interpolatable=True),
                        gt_labels: gp.ArraySpec(voxel_size=(48,48), interpolatable=False),
                    }
                )
                for i in range(0, 10)
            ) +

            gp.RandomProvider() +

            # gp.MapLabels(
            #     gt_labels,
            #     {0: 0, 1: 0, 2: 1, 3: 2, 4: 1}) +

            gp.ExcludeLabels(
                gt_labels,
                [0],
                background_value=1) +

            gp.ExcludeLabels(
                gt_labels,
                [4],
                background_value=2),


            # Labels:
            # 0: background
            # 1: empty
            # 2: bundles
            # 3: tissue
            # 4: neuropil
            # 5: esophagus
            tuple(
                gp.DirectorySource(
                    'data/1018/0002_Anterior',
                    {

                        raw: '{i}.png'.format(i=560+i),
                        gt_labels: 'label_render/ttc/labels000{i}.tif'.format(i=i)
                    },
                    {
                        raw: gp.ArraySpec(voxel_size=(48,48), interpolatable=True),
                        gt_labels: gp.ArraySpec(voxel_size=(48,48), interpolatable=False),
                    }
                )
                for i in range(0, 10)
            ) +

            gp.RandomProvider() +

            gp.ExcludeLabels(
                gt_labels,
                [0],
                background_value=1) +

            gp.ExcludeLabels(
                gt_labels,
                [3, 5],
                background_value=2) +

            gp.ExcludeLabels(
                gt_labels,
                [4],
                background_value=3)
        ) +

        # Labels:
        # 0: background/empty
        # 1: tissue/bundles/esophagus
        # 2: neuropil

        gp.RandomProvider() +

        gp.IntensityScaleShift(
            gt_labels,
            scale=1,
            shift=-1) +

        # convert raw to float in [0, 1]
        gp.Normalize(raw) +

        # chose a random location for each requested batch
        gp.RandomLocation() +

        # apply transpose and mirror augmentations
        gp.SimpleAugment() +

        # scale and shift the intensity of the raw array
        gp.IntensityAugment(
            raw,
            scale_min=0.9,
            scale_max=1.1,
            shift_min=-0.1,
            shift_max=0.1) +

        expand.stubs() +

        expand +

        # create a weight array that balances positive and negative samples in
        # the affinity array
        gp.BalanceLabels(
            gt_labels,
            loss_weights,
            num_classes=NUM_CLASSES) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=1000,
            num_workers=8) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            './train_net',
            net_config['optimizer'],
            net_config['loss'],
            inputs={
                net_config['raw']: raw,
                net_config['gt_labels']: gt_labels,
                net_config['loss_weights']: loss_weights
            },
            outputs={
                # TODO: Mystery
                # Note that for some reason the logits rather than labels
                # must be requested here. If labels are requested, the network
                # only learns class 0.
                net_config['pred_logits']: pred_labels
            },
            gradients={
                net_config['pred_labels_swap']: pred_labels_gradients
            },
            save_every=10000,
            log_dir=os.path.join('log', run_name),
            log_every=100,
            summary=net_config['summary'],
            # array_specs={
            #     pred_labels: gp.ArraySpec(voxel_size=(48,48,1)),
            #     pred_labels_gradients: gp.ArraySpec(voxel_size=())
            # }
            ) +

        gp.Squeeze({
            raw: 0,
            gt_labels: 0,
            loss_weights: 0
        }) +

        # save the passing batch as an HDF5 file for inspection
        # gp.Snapshot(
        #     {
        #         raw: '/volumes/raw',
        #         gt_labels: '/volumes/labels',
        #         pred_labels: '/volumes/pred_labels',
        #         pred_labels_gradients: '/volumes/pred_labels_gradients'
        #     },
        #     output_dir='snapshots',
        #     output_filename='batch_{iteration}.hdf',
        #     every=100,
        #     additional_request=snapshot_request,
        #     compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=100)
    )

    #########
    # TRAIN #
    #########

    print("Training for", iterations, "iterations")

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

    print("Finished")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(2000000, sys.argv[1])

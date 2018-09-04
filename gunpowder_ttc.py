from __future__ import print_function
import gunpowder as gp
import json
import math

def train(iterations):

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
    voxel_size = gp.Coordinate((48, 48))
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape'])*voxel_size

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

    ##############################
    # ASSEMBLE TRAINING PIPELINE #
    ##############################

    pipeline = (

        gp.DirectorySource(
            '/home/championa/data/nadine/1018/sequence_export',
            {
                raw: 'z=0.0 to z=1170000-1.tif',
                gt_labels: 'Labels3.tif'
            },
            {
                raw: gp.ArraySpec(voxel_size=(48,48), interpolatable=True),
                gt_labels: gp.ArraySpec(voxel_size=(48,48), interpolatable=False),
            }
        ) +

        gp.ExcludeLabels(
            gt_labels,
            [0],
            background_value=1) +

        gp.ExcludeLabels(
            gt_labels,
            [4],
            background_value=2) +

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

        # create a weight array that balances positive and negative samples in
        # the affinity array
        gp.BalanceLabels(
            gt_labels,
            loss_weights,
            num_classes=NUM_CLASSES) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=400,
            num_workers=5) +

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
            log_dir='log',
            log_every=100,
            summary=net_config['summary'],
            # array_specs={
            #     pred_labels: gp.ArraySpec(voxel_size=(48,48,1)),
            #     pred_labels_gradients: gp.ArraySpec(voxel_size=())
            # }
            ) +

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
    train(100000)

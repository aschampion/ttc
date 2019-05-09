from __future__ import print_function
import gunpowder as gp
import json
import numpy as np

def predict(iteration):

    ##################
    # DECLARE ARRAYS #
    ##################

    # raw intensities
    raw = gp.ArrayKey('RAW')

    # the predicted affinities
    pred_labels = gp.ArrayKey('PRED_LABELS')

    ####################
    # DECLARE REQUESTS #
    ####################

    with open('test_net_nuclei_config.json', 'r') as f:
        net_config = json.load(f)

    # get the input and output size in world units (nm, in this case)
    voxel_size = gp.Coordinate((48, 48, 48))
    input_size = gp.Coordinate([64,] + net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate([64,] + net_config['output_shape'][1:])*voxel_size
    context = input_size - output_size

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(pred_labels, output_size)

    #############################
    # ASSEMBLE TESTING PIPELINE #
    #############################

    source = gp.N5Source(
            'data/1018/larva-1018.n5',
            {
                raw: '/volumes/raw/c0/s2',
            },
            {
                raw: gp.ArraySpec(voxel_size=(48, 48, 48), interpolatable=True),
            }
#           )
        ) + gp.Crop(
                raw,
                fraction_negative=(0.48, 0, 0),
                fraction_positive=(0.48, 0, 0))

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

    pipeline = (

        # read from HDF5 file
        source +

        # convert raw to float in [0, 1]
        gp.Normalize(raw) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Predict(
            graph='./test_net_nuclei.meta',
            checkpoint='train_net_nuclei_checkpoint_%d'%iteration,
            inputs={
                net_config['raw']: raw
            },
            outputs={
                net_config['pred_labels']: pred_labels
            },
            array_specs={
                pred_labels: gp.ArraySpec(roi=raw_roi.grow(-context, -context), dtype=np.uint32)
            }) +

        # store all passing batches in the same HDF5 file
        gp.N5Write(
            {
                pred_labels: '/volumes/pred_labels',
            },
            output_filename='predictions.n5',
            compression_type='gzip'
            #compression_type='raw'
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

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    predict(2000000)

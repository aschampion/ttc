from gunpowder.zoo.tensorflow import unet, conv_pass
import tensorflow as tf
import argparse
import json

def create_network(input_shape, name, num_classes):

    tf.reset_default_graph()

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=(None,) + input_shape)

    # create a U-Net
    raw_batched = tf.expand_dims(raw, 1)  # Add singleton dimension for channels
    unet_output = unet(raw_batched, num_classes*2, 4, [[3,3],[3,3],[3,3]])

    # add a convolution layer to create 3 output maps representing affinities
    # in z, y, and x
    pred_affs_batched = conv_pass(
        unet_output,
        kernel_size=1,
        num_fmaps=num_classes,
        num_repetitions=1,
        activation=None)

    # get the shape of the output
    output_shape_batched = pred_affs_batched.get_shape().as_list()
    output_shape = [None,] + output_shape_batched[2:] # strip the channels dimension

    # the 4D output tensor (3, depth, height, width)
    # pred_affs_swap = tf.reshape(pred_affs_batched, output_shape_batched[1:])
    pred_affs_swap = tf.transpose(pred_affs_batched, [0, 2, 3, 1])

    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.int32, shape=output_shape)

    # create a placeholder for per-voxel loss weights
    output_shape_broadcast = [1,] + output_shape[1:]
    loss_weights = tf.placeholder_with_default(
        tf.constant(1.0, shape=output_shape_broadcast, dtype=tf.float32),
        shape=output_shape)

    # compute the loss as the weighted mean squared error between the
    # predicted and the ground-truth affinities
    loss = tf.losses.sparse_softmax_cross_entropy(
        gt_affs,
        pred_affs_swap,
        loss_weights)

    # use the Adam optimizer to minimize the loss
    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    # pred_logits = tf.transpose(pred_affs_swap, [0, 3, 1, 2])
    pred_labels = tf.argmax(pred_affs_batched, axis=1, output_type=tf.int32)
    gt_labels_u8 = tf.cast(gt_affs, dtype=tf.uint8)
    pred_labels_u8 = tf.cast(pred_labels, dtype=tf.uint8)

    tf.summary.scalar("loss", loss)
    tf.summary.image("raw", tf.transpose(raw_batched, [0, 2, 3, 1]))
    tf.summary.image("gt_labels", tf.expand_dims(tf.scalar_mul(16, gt_labels_u8), -1))
    tf.summary.image("pred_labels", tf.expand_dims(tf.scalar_mul(16, pred_labels_u8), -1))
    merged_summary_op = tf.summary.merge_all()

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=name + '.meta')

    # store network configuration for use in train and predict scripts
    config = {
        'raw': raw.name,
        'pred_labels_swap': pred_affs_swap.name,
        # 'pred_logits': pred_logits.name,
        'pred_labels': pred_labels.name,
        'pred_labels_u8': pred_labels_u8.name,
        'gt_labels': gt_affs.name,
        'loss_weights': loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'summary': merged_summary_op.name
    }
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make networks for tissue and nuclei segmentation"
    )

    parser.add_argument("--ttc_train_size", type=int, default=268, required=False)
    parser.add_argument("--ttc_num_classes", type=int, default=3, required=False)
    parser.add_argument("--nuclei_train_size", type=int, default=268, required=False)
    parser.add_argument("--test_size", type=int, default=1024, required=False)

    args = parser.parse_args()

    # create a network for training
    create_network((args.ttc_train_size, args.ttc_train_size), 'train_net', args.ttc_num_classes)

    # create a larger network for faster prediction
    create_network((args.test_size, args.test_size), 'test_net', args.ttc_num_classes)

    # create a network for training
    create_network((args.nuclei_train_size, args.nuclei_train_size), 'train_net_nuclei', 2)

    # create a larger network for faster prediction
    create_network((args.test_size, args.test_size), 'test_net_nuclei', 2)

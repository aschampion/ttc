from gunpowder.zoo.tensorflow import unet, conv_pass
import tensorflow as tf
import json

def create_network(input_shape, name, num_classes):

    tf.reset_default_graph()

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=input_shape)

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)
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
    output_shape = output_shape_batched[2:] # strip the batch dimension

    # the 4D output tensor (3, depth, height, width)
    pred_affs_swap = tf.reshape(pred_affs_batched, output_shape_batched[1:])
    pred_affs_swap = tf.transpose(pred_affs_swap, [1, 2, 0])

    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.int32, shape=output_shape)

    # create a placeholder for per-voxel loss weights
    loss_weights = tf.placeholder(
        tf.float32,
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

    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()

    pred_logits = tf.transpose(pred_affs_swap, [2, 0, 1])
    pred_labels = tf.argmax(pred_logits, axis=0)

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=name + '.meta')

    # store network configuration for use in train and predict scripts
    config = {
        'raw': raw.name,
        'pred_labels_swap': pred_affs_swap.name,
        'pred_logits': pred_logits.name,
        'pred_labels': pred_labels.name,
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

    # create a network for training
    create_network((268, 268), 'train_net', 3)

    # create a larger network for faster prediction
    create_network((322, 322), 'test_net', 3)

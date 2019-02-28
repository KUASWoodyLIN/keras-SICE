import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import conv_utils


class Conv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', align_shape=None, **kwargs):
        super(Conv2DTranspose, self).__init__(filters, kernel_size, strides, padding, **kwargs)
        if type(align_shape) != type(None):
            self.align_shape = tf.shape(align_shape)[1:3]
        else:
            self.align_shape = None

    def call(self, inputs, **kwargs):
        inputs_shape = tf.shape(inputs)
        # align_shape = tf.shape(inputs[1])
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        if type(self.align_shape) == type(None):
            out_height = conv_utils.deconv_output_length(height,
                                                         kernel_h,
                                                         self.padding,
                                                         stride_h)
            out_width = conv_utils.deconv_output_length(width,
                                                        kernel_w,
                                                        self.padding,
                                                        stride_w)
        else:
            out_height = self.align_shape[0]
            out_width = self.align_shape[1]
        # out_height = tf.where(tf.equal(out_height % 2, 0), out_height, out_height+1)
        # out_width = tf.where(tf.equal(out_width % 2, 0), out_width, out_width+1)

        # out_height = align_shape[1]
        # out_width = align_shape[2]

        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
            strides = (1, 1, stride_h, stride_w)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            strides = (1, stride_h, stride_w, 1)

        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = inputs.get_shape().as_list()
            out_shape[c_axis] = self.filters
            out_shape[h_axis] = conv_utils.deconv_output_length(out_shape[h_axis],
                                                                kernel_h,
                                                                self.padding,
                                                                stride_h)
            out_shape[w_axis] = conv_utils.deconv_output_length(out_shape[w_axis],
                                                                kernel_w,
                                                                self.padding,
                                                                stride_w)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    # def compute_output_shape(self, input_shape):
    #     input_shape = tf.TensorShape(input_shape).as_list()
    #     output_shape = list(input_shape)
    #     if self.data_format == 'channels_first':
    #         c_axis, h_axis, w_axis = 1, 2, 3
    #     else:
    #         c_axis, h_axis, w_axis = 3, 1, 2
    #
    #     kernel_h, kernel_w = self.kernel_size
    #     stride_h, stride_w = self.strides
    #
    #     output_shape[c_axis] = self.filters
    #
    #     out_height = conv_utils.deconv_output_length(output_shape[h_axis], kernel_h, self.padding, stride_h)
    #     out_width = conv_utils.deconv_output_length(output_shape[w_axis], kernel_w, self.padding, stride_w)
    #     output_shape[h_axis] = tf.where(tf.equal(out_height % 2, 0), out_height, out_height+1)
    #     output_shape[w_axis] = tf.where(tf.equal(out_width % 2, 0), out_width, out_width+1)
    #     return tf.TensorShape(output_shape)

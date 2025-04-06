#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>

#include "aifes.h"
#include "MNIST_training_data.h"
#include "MNIST_training_data_label.h"
#include "MNIST_test_data.h"
#include "MNIST_test_data_label.h"

#define NUM_TRAIN_SAMPLES  1000
#define NUM_TEST_SAMPLES   200
#define IMG_HEIGHT         28
#define IMG_WIDTH          28
#define IMG_CHANNELS       1
#define NUM_CLASSES        10

int main(int argc, char *argv[]) {
    uint32_t i;
    time_t t;
    srand((unsigned) time(&t));
    printf("MNIST CNN Classification:\n");

    float *input_data = (float *)MNIST_training_data;
    float *target_data = (float *)MNIST_training_data_label;
    float *test_data = (float *)MNIST_test_data;
    float *test_target_data = (float *)MNIST_test_data_label;

    uint16_t input_shape[] = {NUM_TRAIN_SAMPLES, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH};
    uint16_t target_shape[] = {NUM_TRAIN_SAMPLES, NUM_CLASSES};
    uint16_t test_shape[] = {NUM_TEST_SAMPLES, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH};
    uint16_t test_target_shape[] = {NUM_TEST_SAMPLES, NUM_CLASSES};

    aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, input_data);
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);
    aitensor_t test_tensor = AITENSOR_4D_F32(test_shape, test_data);
    aitensor_t test_target_tensor = AITENSOR_2D_F32(test_target_shape, test_target_data);

    // ----------------- CNN Model -----------------
    uint16_t input_layer_shape[] = {20, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH};
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_layer_shape);

    ailayer_conv2d_t conv2d_layer_1 = AILAYER_CONV2D_F32_A(8, HW(3, 3), HW(1, 1), HW(1, 1), HW(0, 0));
    ailayer_batch_norm_f32_t bn_layer_1 = AILAYER_BATCH_NORM_F32_A(0.9f, 1e-6f);
    ailayer_relu_f32_t relu_layer_1 = AILAYER_RELU_F32_A();
    ailayer_maxpool2d_t maxpool2d_layer_1 = AILAYER_MAXPOOL2D_F32_A(HW(2, 2), HW(2, 2), HW(0, 0));

    ailayer_conv2d_t conv2d_layer_2 = AILAYER_CONV2D_F32_A(16, HW(3, 3), HW(1, 1), HW(1, 1), HW(0, 0));
    ailayer_batch_norm_f32_t bn_layer_2 = AILAYER_BATCH_NORM_F32_A(0.9f, 1e-6f);
    ailayer_relu_f32_t relu_layer_2 = AILAYER_RELU_F32_A();
    ailayer_maxpool2d_t maxpool2d_layer_2 = AILAYER_MAXPOOL2D_F32_A(HW(2, 2), HW(2, 2), HW(0, 0));

    ailayer_conv2d_t conv2d_layer_3 = AILAYER_CONV2D_F32_A(32, HW(3, 3), HW(1, 1), HW(1, 1), HW(0, 0));
    ailayer_relu_f32_t relu_layer_3 = AILAYER_RELU_F32_A();
    ailayer_maxpool2d_t maxpool2d_layer_3 = AILAYER_MAXPOOL2D_F32_A(HW(2, 2), HW(2, 2), HW(0, 0));

    ailayer_flatten_t flatten_layer = AILAYER_FLATTEN_F32_A();
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(64);
    ailayer_relu_f32_t relu_layer_4 = AILAYER_RELU_F32_A();
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(NUM_CLASSES);
    ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_A();

    ailoss_crossentropy_f32_t crossentropy_loss;

    // ----------------- Model Structure -----------------
    aimodel_t model;
    ailayer_t *x;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_conv2d_chw_f32_default(&conv2d_layer_1, model.input_layer);
    x = ailayer_batch_norm_cfirst_f32_default(&bn_layer_1, x);
    x = ailayer_relu_f32_default(&relu_layer_1, x);
    x = ailayer_maxpool2d_chw_f32_default(&maxpool2d_layer_1, x);

    x = ailayer_conv2d_chw_f32_default(&conv2d_layer_2, x);
    x = ailayer_batch_norm_cfirst_f32_default(&bn_layer_2, x);
    x = ailayer_relu_f32_default(&relu_layer_2, x);
    x = ailayer_maxpool2d_chw_f32_default(&maxpool2d_layer_2, x);

    x = ailayer_conv2d_chw_f32_default(&conv2d_layer_3, x);
    x = ailayer_relu_f32_default(&relu_layer_3, x);
    x = ailayer_maxpool2d_chw_f32_default(&maxpool2d_layer_3, x);

    x = ailayer_flatten_f32_default(&flatten_layer, x);
    x = ailayer_dense_f32_default(&dense_layer_1, x);
    x = ailayer_relu_f32_default(&relu_layer_4, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_softmax_f32_default(&softmax_layer, x);
    model.output_layer = x;

    model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);

    aialgo_compile_model(&model);

    printf("-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    printf("----------------------------------------------\n\n");

    // ---------------- Memory Allocation ----------------
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    printf("Required memory for parameter (Weights, Bias, ...): %u Byte\n", parameter_memory_size);
    void *parameter_memory = malloc(parameter_memory_size);
    if (!parameter_memory) {
        fprintf(stderr, "Error: Failed to allocate memory for parameters.\n");
        exit(1);
    }
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);
    aialgo_initialize_parameters_model(&model);

    // ---------------- Optimizer ----------------
    aiopti_t *optimizer;
    aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.01f, 0.9f, 0.999f, 1e-6f);
    optimizer = aiopti_adam_f32_default(&adam_opti);

    // ------------- Training Memory Allocation -------------
    uint32_t working_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    printf("Required memory for training (Intermediate result, gradients, momentums): %u Byte\n", working_memory_size);
    void *working_memory = malloc(working_memory_size);
    if (!working_memory) {
        fprintf(stderr, "Error: Failed to allocate memory for training.\n");
        free(parameter_memory);
        exit(1);
    }
    aialgo_schedule_training_memory(&model, optimizer, working_memory, working_memory_size);
    aialgo_init_model_for_training(&model, optimizer);

    // ------------------ Training ------------------
    float loss;
    uint32_t batch_size = 200;
    uint32_t epochs = 150;

    printf("Start training.\n");
    for (i = 0; i < epochs; i++) {
        aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);
        if (i % 10 == 0) {
            aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
            printf("Epoch %u: Loss = %f\n", i, loss);
        }
    }

    // ------------------ Evaluation ------------------
    aitensor_t test_input_tensor = AITENSOR_4D_F32(test_shape, input_data);

    float output_data[200 * NUM_CLASSES];  // adjust to batch size * classes
    uint16_t output_shape[2] = {200, NUM_CLASSES};
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    printf("\nResults after training:\n");

    aialgo_inference_model(&model, &test_input_tensor, &output_tensor);
    aialgo_inference_model(&model, &test_tensor, &output_tensor);

    uint32_t correct_predictions = 0;
    for (uint32_t sample = 0; sample < NUM_TEST_SAMPLES; ++sample) {
        uint32_t predicted_class = 0;
        float max_prob = output_data[sample * NUM_CLASSES];

        // Find predicted class
        for (uint32_t j = 1; j < NUM_CLASSES; ++j) {
            float prob = output_data[sample * NUM_CLASSES + j];
            if (prob > max_prob) {
                max_prob = prob;
                predicted_class = j;
            }
        }

        // Find true class
        uint32_t true_class = 0;
        for (uint32_t j = 0; j < NUM_CLASSES; ++j) {
            if (test_target_data[sample * NUM_CLASSES + j] == 1.0f) {
                true_class = j;
                break;
            }
        }

        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / NUM_TEST_SAMPLES * 100.0f;
    printf("\nFinal Test Accuracy: %.2f%%\n", accuracy);


    free(working_memory);
    free(parameter_memory);
    return 0;
}

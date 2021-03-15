import models
import datasets
from regularization_helps import finite_data_experiment, regularization_results, rotated_mnist_regularization_experiment


def rotated_mnist_60_conv_experiment_ablation():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_rot_mnist_60_conv_ablation.dat',
        unreg_model_func=models.simple_softmax_conv_model_bn,
        reg_model_func=models.simple_softmax_conv_model_dropout,
        interval=2000, epochs=10, loss='ce', soft=False, num_runs=5)


def soft_rotated_mnist_60_conv_experiment_ablation():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_soft_rot_mnist_60_conv_ablation.dat',
        unreg_model_func=models.simple_softmax_conv_model_bn,
        reg_model_func=models.simple_softmax_conv_model_dropout,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def retrain_soft_rotated_mnist_60_conv_experiment():
    finite_data_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/reg_vs_unreg_retrain_soft_rot_mnist_60_conv_ablation.dat',
        unreg_model_func=models.simple_softmax_conv_model_bn,
        reg_model_func=models.simple_softmax_conv_model_dropout,
        interval=2000, epochs=10, loss='categorical_ce', soft=True, num_runs=5)


def portraits_conv_experiment_ablation():
    finite_data_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/reg_vs_unreg_portraits_ablation.dat',
        unreg_model_func=models.simple_softmax_conv_model_bn,
        reg_model_func=models.simple_softmax_conv_model_dropout,
        interval=2000, epochs=20, loss='ce', soft=False, num_runs=5)


def soft_portraits_conv_experiment_ablation():
    finite_data_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/reg_vs_unreg_soft_portraits_ablation.dat',
        unreg_model_func=models.simple_softmax_conv_model_bn,
        reg_model_func=models.simple_softmax_conv_model_dropout,
        interval=2000, epochs=20, loss='categorical_ce', soft=True, num_runs=5)


if __name__ == "__main__":
    rotated_mnist_regularization_experiment(
        models.simple_softmax_conv_model_bn, models.simple_softmax_conv_model_dropout, 'ce',
        save_name_base='saved_files/inf_reg_mnist_ablation', N=2000, delta_angle=3, num_angles=20,
        retrain=False, num_runs=5)
    print("Rot MNIST experiment 2000 points rotated")
    regularization_results('saved_files/inf_reg_mnist_ablation_2000_3_20.dat')

    rotated_mnist_regularization_experiment(
        models.simple_softmax_conv_model_bn, models.simple_softmax_conv_model_dropout, 'ce',
        save_name_base='saved_files/inf_reg_mnist', N=5000, delta_angle=3, num_angles=20,
        retrain=False, num_runs=5)
    print("Rot MNIST experiment 5000 points rotated")
    regularization_results('saved_files/inf_reg_mnist_ablation_5000_3_20.dat')

    rotated_mnist_regularization_experiment(
        models.simple_softmax_conv_model_bn, models.simple_softmax_conv_model_dropout, 'ce',
        save_name_base='saved_files/inf_reg_mnist', N=20000, delta_angle=3, num_angles=20,
        retrain=False, num_runs=5)
    print("Rot MNIST experiment 20k points rotated")
    regularization_results('saved_files/inf_reg_mnist_ablation_20000_3_20.dat')

    # Run all experiments comparing regularization vs no regularization.
    portraits_conv_experiment_ablation()
    print("Portraits conv experiment reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_portraits_ablation.dat')
    rotated_mnist_60_conv_experiment_ablation()
    print("Rotating MNIST conv experiment reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_rot_mnist_60_conv_ablation.dat')

    # Run all experiments, soft labeling, comparing regularization vs no regularization.
    soft_portraits_conv_experiment_ablation()
    print("Portraits conv experiment soft labeling reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_soft_portraits_ablation.dat')
    soft_rotated_mnist_60_conv_experiment_ablation()
    print("Rot MNIST conv experiment soft labeling reg vs no reg")
    regularization_results('saved_files/reg_vs_unreg_soft_rot_mnist_60_conv_ablation.dat')

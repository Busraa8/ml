# %% 
if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# %%
@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    #raise NotImplementedError("Your Code Goes Here")
    
    print('Running 10 clusters')
    centers_10 = lloyd_algorithm(x_train,10)
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(centers_10[i].reshape((28, 28)))
    plt.suptitle('Visualize 10 centers')
    plt.show()

    k_vals = [2, 4, 8, 16, 32, 64]
    train_errors = np.zeros(len(k_vals))
    test_errors = np.zeros(len(k_vals))

    for i, k in enumerate(k_vals):
        print(f"Running {k} clusters")
        centers = lloyd_algorithm(x_train, k)
        train_errors[i] = calculate_error(x_train, centers)
        test_errors[i] = calculate_error(x_test, centers)

    plt.plot(k_vals, train_errors, "o-b", label="Train")
    plt.plot(k_vals, test_errors, "o-r", label="Test")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    main()

# %%
#(x_train, _), (x_test, _) = load_dataset("mnist")

# f, axarr = plt.subplots(2,5)
# for i in range(10):
#     axarr[i%2,i%5].imshow(centers_10[i].reshape((28,28)))
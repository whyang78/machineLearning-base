import matplotlib.pyplot as plt
def plot_curve(param_sizes,cv_result,xlabel):
    train_score_mean=cv_result['mean_train_score']
    train_score_std=cv_result['std_train_score']
    test_score_mean=cv_result['mean_test_score']
    test_score_std=cv_result['std_test_score']

    plt.figure()
    plt.title('parameters turning')
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.grid()

    plt.fill_between(param_sizes,train_score_mean-train_score_std,
                     train_score_mean+train_score_std,color='r',alpha=0.2)
    plt.fill_between(param_sizes,test_score_mean-test_score_std,
                     test_score_mean+test_score_std,color='g',alpha=0.2)
    plt.plot(param_sizes,train_score_mean,'.--',color='r',label='train_scores')
    plt.plot(param_sizes,test_score_mean,'.-',color='g',label='Cross_validation_scores')
    plt.legend(loc='best')

    plt.show()


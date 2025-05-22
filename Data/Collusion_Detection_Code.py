import os
import sys
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import warnings
from joblib import dump, load
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from collections import defaultdict
from scipy import stats
from itertools import cycle


# Font size to plot
default_font_size = 18
plt.rcParams.update({'font.size': default_font_size})

# Format to print
pd.options.display.float_format = '{:,.4f}'.format

# To hide warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths and filenames of the datasets
path = os.path.dirname(os.path.abspath(__file__))
db_collusion_brazilian = os.path.join(path, 'DB_Collusion_Brazil_processed.csv')
db_collusion_italian = os.path.join(path, 'DB_Collusion_Italy_processed.csv')
db_collusion_american = os.path.join(path, 'DB_Collusion_America_processed.csv')
db_collusion_switzerland_gr_sg = os.path.join(path, 'DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv')
db_collusion_switzerland_ticino = os.path.join(path, 'DB_Collusion_Switzerland_Ticino_processed.csv')
db_collusion_japan = os.path.join(path, 'DB_Collusion_Japan_processed.csv')
db_collusion_all = os.path.join(path, 'DB_Collusion_All_processed.csv')

# To save plots (pdf format)
plot_pdf = True 

# User's parameters for the functions
ml_algorithms = ['GaussianProcessClassifier', 'SGDClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 
                  'GradientBoostingClassifier', 'SVC', 'KNeighborsClassifier', 'MLPClassifier', 'BernoulliNB', 'GaussianNB'] 
screens = ['CV', 'SPD', 'DIFFP', 'RD', 'KURT', 'SKEW', 'KSTEST'] # Screening variables to use. There are seven: CV, SPD, DIFFP, RD, KURT, SKEW and KSTEST 
train_size = 0.8 # Test and train sizes. The test_size is 1-train_size
repetitions = 50 # Number of repetitions for each ML algorithm. Minimum value > 30. Recommended value > 100
n_estimators = 300 # Number of estimators for ML algorithms
precision_recall = True # To plot precision-recall curves
load_data = False # To load the error metrics (to load previous data experimentation)
save_data = True # To save the error metrics (to persist the data experimentation)


def shuffle_tenders(df1):
    ''' Shuffle tenders. The reason is that maybe the colluded tenders are concentrated in some parts of the excel (dataframe)'''

    df = df1.copy()
    df = df.sample(frac=1).reset_index(drop=True)
    df['Tender'] = df['Tender'].astype(str)
    reindex_tenders = 1
    list_tenders = []
    for index, row in df.iterrows():
        if not row['Tender'] in list_tenders:
            df['Tender'].replace(row['Tender'], reindex_tenders, inplace=True)
            reindex_tenders = reindex_tenders + 1
            list_tenders.append(row['Tender'])
    return df


def calculate_colluded_tenders_by_bidder(df):
    ''' Calculate the colluded tenders by bidder and print the results '''

    df_aux = df.copy()
    df_tenders_by_bidder = df_aux.groupby(['Competitors']).size().reset_index(name='Total_tenders')
    df_aux['Collusive_competitor'] = df_aux['Collusive_competitor'].apply(lambda x: 1 if x > 0 else x)
    df_tenders_by_bidder['Total_colluded_tenders'] = df_aux.groupby(['Competitors'])['Collusive_competitor'].sum()
    df_tenders_by_bidder['Ratio'] = df_tenders_by_bidder['Total_colluded_tenders'] / df_tenders_by_bidder['Total_tenders'] * 100
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_tenders_by_bidder)


def printScatterMatrix(df, color_labels, colors, labels_legend, dataset):
    ''' Scatter matrix for a dataframe '''

    plt.rcParams.update({'font.size': 11}) # Font size to plot
    sm = pd.plotting.scatter_matrix(df, figsize=(12, 16), diagonal='kde', alpha=0.35, color=color_labels, s=3, rasterized=True) # kde or hist
    n = len(df.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = sm[x, y]
            # to make x axis name vertical  
            #ax.xaxis.label.set_rotation(330)
            # to make y axis name horizontal 
            #ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            #ax.yaxis.labelpad = 40
            ax.xaxis.labelpad = 20
            # to show the half of the scatter matrix
            if x < y:
                sm[x, y].set_visible(False)
            # adjust xlim, ylim
            if x == 1:
                ax.set_ylim([0, 0.4])
            elif x == 2:
                ax.set_ylim([0, 0.5])
            elif x == 3:
                ax.set_ylim([0, 0.4])
            elif x == 4:
                ax.set_ylim([-750, 750])
            elif x == 5:
                ax.set_ylim([-5, 10])
            elif x == 6:
                ax.set_ylim([-3, 3])
            if y == 0:
                ax.set_xlim([0, 0.2 * 1000000000])
            elif y == 1:
                ax.set_xlim([0, 0.4])
            elif y == 2:
                ax.set_xlim([0, 0.5])
            elif y == 3:
                ax.set_xlim([0, 0.4])
            elif y == 4:
                ax.set_xlim([-750, 750])
            elif y == 5:
                ax.set_xlim([-5, 10])    
            elif y == 6:
                ax.set_xlim([-3, 3])
    # More bottom margin to read x and y axis
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # Legend
    handles = [plt.plot([], [], color=colors[i], ls='', marker='.', markersize=np.sqrt(90))[0] for i in range(len(colors))]
    plt.legend(handles, labels_legend, loc=(-1,4)) 
    # Draw
    plt.draw()
    if plot_pdf:
        name_file = dataset + '_Scatter_matrix.pdf'
        plt.savefig(name_file, format='pdf', dpi=1200, bbox_inches='tight')
        print('Generated and saved file called ' + name_file)
    plt.rcParams.update({'font.size': default_font_size}) # Font size to plot


def print_boxplot(df, dataset, column_names, groupby, min_ylim, max_ylim, step_y, xlabel, percentage=True):
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(2*0.7,6*0.7))
    ax = fig.gca()
    df.boxplot(column=column_names, by=groupby, ax=ax, fontsize=None, rot=0, grid=False, notch=True, widths=0.3, positions = (0.3, 0.9),
                            layout=None, return_type=None, showfliers=False, meanline=True, showmeans=True, patch_artist=True, vert=True,
                            medianprops=dict(linestyle='-', linewidth=3, color='limegreen'), meanprops=dict(linestyle='-', linewidth=3, color='firebrick'))
    # Configurate plotting
    ax.set_title(column_names.replace('_', ' '), fontweight='bold')
    fig.suptitle('')
    plt.xlabel(xlabel)
    ax.set_ylim([min_ylim, max_ylim])
    ax.set_yticks(np.arange(min_ylim, max_ylim+step_y, step_y))
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    if percentage:
        ax.set_yticklabels(['{:.2f}%'.format(x) for x in plt.gca().get_yticks()]) # Percentage format 
    custom_lines = [Line2D([0], [0], color='firebrick', lw=3),
                    Line2D([0], [0], color='limegreen', lw=3)]
    names_lines = ['Mean', 'Median']
    ax.legend(custom_lines, names_lines, loc='center', bbox_to_anchor=(1.75, 0.125))
    plt.draw()
    if plot_pdf:
        name_file = dataset + '_BoxPlot_' + column_names + '.pdf'
        fig.savefig(name_file, format='pdf', dpi=1200, bbox_inches='tight')
        print('Generated and saved file called ' + name_file)
    plt.rcParams.update({'font.size': default_font_size})


def predict_collusion_company(df, dataset, predictors_column_name, targets_column_name, algorithm, train_size, n_estimators=None):
    ''' Predict collusion applying the ML algorithm '''

    # Datasets to have to simplify the process' time 
    simplify_process = ['japan', 'italian', 'switzerland_gr_sg', 'american', 'all']

    # To assing the dataframes
    predictors = df[predictors_column_name]
    targets = df[targets_column_name]

    # We create the training and test sample, both for predictors and for the objective variable, based on the tender group. 
    # That is, the bids of a tender either all own to the train group or the test group. They cannot be divided between both groups. 
    gss = GroupShuffleSplit(n_splits=5, train_size=train_size)
    train_index, test_index = next(gss.split(predictors, targets, groups=df['Tender']))
    x_train = predictors.loc[train_index]
    y_train = targets.loc[train_index]
    x_test = predictors.loc[test_index]
    y_test = targets.loc[test_index]

    # Train the model with the selected algorithm
    if algorithm == 'ExtraTreesClassifier':
        classifier = ExtraTreesClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                            oob_score=True, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight='balanced', ccp_alpha=0.0, max_samples=None)
    elif algorithm == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., 
                            max_features=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, 
                            oob_score=True, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight='balanced')
    elif algorithm == 'SGDClassifier':
        classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=10000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, 
                            n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
                            class_weight=None, warm_start=False, average=False)
    elif algorithm == 'AdaBoostClassifier':
        classifier = AdaBoostClassifier(base_estimator=None, n_estimators=n_estimators, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    elif algorithm == 'GradientBoostingClassifier':
        if dataset in simplify_process:
            learning_rate = 100
            tol = 10
            estimators = int(round(n_estimators / 3))
        else:
            learning_rate = 0.1
            tol = 0.0001
            estimators = n_estimators
        classifier = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate, n_estimators=estimators, subsample=1.0, criterion='mae', min_samples_split=2, min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, max_depth=None, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, 
                            max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=tol, ccp_alpha=0.0)
    elif algorithm == 'SVC':
        classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
                            class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    elif algorithm == 'KNeighborsClassifier':
        classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    elif algorithm == 'MLPClassifier':
        classifier = MLPClassifier(hidden_layer_sizes=(240, 120, 70, 35), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                        power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=0, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    elif algorithm == 'GaussianNB':
        classifier = GaussianNB(priors=None, var_smoothing=1e-09)
    elif algorithm == 'BernoulliNB':
        classifier = BernoulliNB(alpha=0.5, binarize=0, fit_prior=True, class_prior=None)
    elif algorithm == 'GaussianProcessClassifier':
        if dataset in simplify_process:
            max_iter_predict = 5
            n_restarts_optimizer = 2
        else:
            max_iter_predict = 5000
            n_restarts_optimizer = 50
        classifier = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=n_restarts_optimizer, max_iter_predict=max_iter_predict, warm_start=False, copy_X_train=True, random_state=None, 
                        multi_class='one_vs_rest', n_jobs=-1)
    
    # We build the model for the train group
    classifier = classifier.fit(x_train, y_train.values.ravel())
  
    # We predict for the values of the test group
    predictions = classifier.predict(x_test)
    df_predictions = pd.DataFrame(data=predictions, index=y_test.index, columns=['Forecast_collusive_competitor'])
    
    # To calculate the error metrics for the classification binary model
    accuracy = accuracy_score(y_test, predictions) * 100
    balanced_accuracy = balanced_accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions, pos_label=1, average='binary', zero_division=1) * 100 # Ratio of true positives: tp / (tp + fp)
    recall = recall_score(y_test, predictions, pos_label=1, average='binary', zero_division=1) * 100 # Ratio of true positives: tp / (tp + fn)
    f1 = f1_score(y_test, predictions, pos_label=1, average='binary', zero_division=1) * 100 # F1 = 2 * (precision * recall) / (precision + recall)
    confusion = confusion_matrix(y_test, predictions, normalize='all') * 100
    
    return accuracy, balanced_accuracy, precision, recall, f1, confusion, y_test, df_predictions


def algorithm_comparison(df, dataset, predictors, targets, algorithms, train_size, repetitions, n_estimators, precision_recall=False, load_data=False, save_data=False):
    ''' Print table to compare Machine Learning algorithms '''

    df = shuffle_tenders(df)
    
    for setting in predictors:
        print('')
        print('Generating models for ' + setting)
        accuracy = defaultdict(list)
        balanced_accuracy = defaultdict(list)
        false_positive = defaultdict(list)
        false_negative = defaultdict(list)
        precision = defaultdict(list)
        recall = defaultdict(list)
        f1 = defaultdict(list)
        tenders_test = defaultdict(list)
        tenders_predictions = defaultdict(list)

        # Create namefile
        namefile = dataset + '_ML_algorithms_experimentation_' + setting + '_' + str(repetitions) + 'repetitions'
        
        if load_data == False:
            for algorithm in algorithms:
                print('Training algorithm ' + algorithm)
                df_copy = df.copy()
                if algorithm in ['GaussianProcessClassifier', 'GradientBoostingClassifier', 'SVC']:
                    loop = int(round(repetitions / 40))
                    if dataset == 'all' and algorithm == 'GaussianProcessClassifier':
                        # Exception: reduce the datataset to be able to compute this dataset and algorithm
                        df_copy = df_copy.sample(frac=0.5).reset_index(drop=True)
                else:
                    loop = repetitions
                for i in range(loop):
                    item_accuracy, item_balanced_accuracy, item_precision, item_recall, item_f1, confusion_matrix, item_tenders_test, item_tenders_predictions = \
                                predict_collusion_company(df_copy, dataset, predictors[setting], targets, algorithm, train_size, n_estimators)
                    accuracy[algorithm].append(item_accuracy)
                    balanced_accuracy[algorithm].append(item_balanced_accuracy)
                    if confusion_matrix.shape[1] == 2:
                        false_positive[algorithm].append(confusion_matrix[0][1])
                        false_negative[algorithm].append(confusion_matrix[1][0])
                    else:
                        false_positive[algorithm].append(0)
                        false_negative[algorithm].append(0)
                    precision[algorithm].append(item_precision)
                    recall[algorithm].append(item_recall)
                    f1[algorithm].append(item_f1)
                    tenders_test[algorithm].append(item_tenders_test)
                    tenders_predictions[algorithm].append(item_tenders_predictions)

            # Save dictionaries to persist the data experimentation 
            if save_data:
                path_namefile = os.path.join(path, namefile + '.pkl')
                file = [accuracy, balanced_accuracy, false_positive, false_negative, precision, recall, f1, df, tenders_test, tenders_predictions]
                dump(file, path_namefile, compress=6)
               
        else:
            # To load data
            pkl_file = os.path.join(path, namefile + '.pkl')
            [accuracy, balanced_accuracy, false_positive, false_negative, precision, recall, f1, df, tenders_test, tenders_predictions] = load(pkl_file)
        
        for algorithm in algorithms:
            # Print error metrics
            test_size = 1 - train_size
            print('Algorithm {} with train:test {:,.2f}:{:,.2f}, {} repetitions and {}: mean_accuracy={:,.1f}, mean_FP={:,.1f}, ' 
                'mean_FN={:,.1f}, mean_balanced_accuracy={:,.1f}, mean_f1={:,.1f}, median_f1={:,.1f}, mean_precision={:,.1f}, '
                'median_precision={:,.1f}, mean_recall={:,.1f} and median_recall={:,.1f}'.format(
                algorithm, train_size, test_size, repetitions, setting, np.mean(accuracy[algorithm]), np.mean(false_positive[algorithm]), np.mean(false_negative[algorithm]), 
                np.mean(balanced_accuracy[algorithm]), np.mean(f1[algorithm]), np.median(f1[algorithm]), np.mean(precision[algorithm]),
                np.median(precision[algorithm]), np.mean(recall[algorithm]), np.median(recall[algorithm])))
        
        # Print curve precision vs recall with iso-F1 lines
        if precision_recall:
            plot_precision_vs_recall(dataset, algorithms, precision, recall, min_f1=0.4, max_f1=0.86, f1_curves=24, min_x_y_lim=0.5, max_x_y_lim=1, namefile=namefile)


def plot_precision_vs_recall(dataset, algorithms, precision, recall, min_f1, max_f1, f1_curves, min_x_y_lim, max_x_y_lim, namefile=None):
    ''' Plot the precision vs recall with F1 Score iso-curves to compare the ML algorithms. 
        The point to cut both lines (precision and recall) is median of the F1 score. 
        This is necessary to correspond the point with the F1 Score iso-curves'''

    plt.rcParams.update({'font.size': 26}) # Font size to plot

    # Colors and markers for the plot to compare 11 algorithms
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                    'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'gold'])
    markers = cycle(['o', '.', 'v', '^', '<', '>', 's', 'p', 'D', 'P', 'X'])

    fig = plt.figure(figsize=(12, 12))
    f1_scores = np.linspace(min_f1, max_f1, num=f1_curves)
    lines = []
    labels = []

    # Create iso-F1 curves
    for f1_scores in f1_scores:
        x = np.linspace(0.01, 1)
        y = f1_scores * x / (2 * x - f1_scores)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.65, linestyle='-.', lw=2)
        plt.annotate('F1={0:0.2f}'.format(f1_scores), xy=(0.94, y[46] - 0.001), fontsize=17)
    lines.append(l)
    labels.append('F1 curves')

    # Convert to [0, 1]
    for item in algorithms:
        recall[item] =  [x / 100 for x in recall[item]]
        precision[item] =  [x / 100 for x in precision[item]]

    # Calculate the points to plot the two lines
    line_precision_x = defaultdict(list)
    line_precision_y = defaultdict(list)
    line_recall_x = defaultdict(list)
    line_recall_y = defaultdict(list)
    for item in algorithms:
        line_recall_x[item] = [np.percentile(recall[item], 25), np.percentile(recall[item], 75)]
        line_recall_y[item] = [np.median(precision[item]), np.median(precision[item])]
        line_precision_x[item] = [np.median(recall[item]), np.median(recall[item])]
        line_precision_y[item] = [np.percentile(precision[item], 25), np.percentile(precision[item], 75)]
    
    # Plot the two lines and the point to cut both lines
    for item, color in zip(algorithms, colors): # It can possible to use markers list
        l, = plt.plot(line_precision_x[item], line_precision_y[item], color=color, lw=4, marker='_', markersize=14, markeredgewidth=4)
        l, = plt.plot(line_recall_x[item], line_recall_y[item], color=color, lw=4, marker='|', markersize=14, markeredgewidth=4)
        l, = plt.plot(np.median(recall[item]), np.median(precision[item]), color=color, markersize=8, marker='o')
        lines.append(l)
        labels.append('{}'.format(item))

    plt.xlim([min_x_y_lim, max_x_y_lim])
    plt.ylim([min_x_y_lim, max_x_y_lim])
    plt.grid(dashes=(5, 10), linewidth=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), prop=dict(size=default_font_size - 2), ncol=3)

    # Axis in percentage format
    ax = fig.gca()
    ax.set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()]) 
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    
    plt.draw()
    if plot_pdf:
        name_file = dataset + '_Precision_Recall_' + namefile + '.pdf'
        fig.savefig(name_file, format='pdf', dpi=1200, bbox_inches='tight')
        print('Generated and saved file called ' + name_file)


def plotTwoHistograms(data_1, data_2, label_1, label_2, max_range, bins, max_xlim, density=True):
    ''' Plot two histograms or density functions '''

    # Fit lognormal distribution
    data_1 = sorted(data_1.values)
    data_2 = sorted(data_2.values)
    shape, loc, scale = stats.lognorm.fit(data_1, loc=0)
    data_1_prob_density_function_lognorm = stats.lognorm.pdf(data_1, shape, loc, scale)
    shape, loc, scale = stats.lognorm.fit(data_2, loc=0)
    data_2_prob_density_function_lognorm = stats.lognorm.pdf(data_2, shape, loc, scale)

    # Plot histograms and density distributions
    fig = plt.figure(figsize=(16, 12))
    plt.hist(data_1, bins=bins, range=(0, max_range), alpha=0.3, label='Histogram: ' + label_1, facecolor='g', density=density)
    plt.hist(data_2, bins=bins, range=(0, max_range), alpha=0.3, label='Histogram: ' + label_2, facecolor='r', density=density)
    plt.plot(data_1, data_1_prob_density_function_lognorm, label='Probability density function (log normal): ' + label_1, color='g', linewidth=3)
    plt.plot(data_2, data_2_prob_density_function_lognorm, label='Probability density function (log normal): ' + label_2, color='r', linewidth=3)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.ylabel('Probability')
    plt.xlabel('Number of bids by tender')
    plt.xlim(0, max_xlim)
    plt.xticks(np.arange(0, max_xlim, step=max_xlim/bins))
    plt.grid(True, linestyle='--', alpha=0.5)

    # Axis in percentage format
    ax = fig.gca()
    ax.set_yticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_yticks()])

    if plot_pdf:
        name_file = dataset + '_Two_density_plots.pdf'
        plt.savefig(name_file, format='pdf', dpi=1200, bbox_inches='tight')
        print('Generated and saved file called ' + name_file)


def print_description_processed_dataset(df):
    ''' Print the most important information for the collusive dataset '''

    # General 
    df_tenders = df.drop_duplicates(subset=['Tender', 'Number_bids'])
    number_tenders = len(df['Tender'].unique())
    print('')
    print('------------------------------------')
    print('Information of the collusive dataset')
    print('Tenders: {0}'.format(number_tenders))
    if 'Date' in df:
        df_aux = df[df['Date'] > 0] # To avoid unavailable timestamps
        minimum_date = datetime.datetime.fromtimestamp(df_aux['Date'].min())
        maximum_date = datetime.datetime.fromtimestamp(df_aux['Date'].max())
        print('Temporal range: {0}, {1}'.format(minimum_date, maximum_date))
    number_bids = len(df)
    print('Bids: {0}'.format(number_bids))
    if 'Collusive_competitor_original' in df:
        column_name = 'Collusive_competitor_original'
    else:
        column_name = 'Collusive_competitor'
    number_collusive_bidders = len(df[df[column_name] == 1])
    mean_number_bids = np.mean(df_tenders['Number_bids'])
    print('Mean value of bidders per tender: {:,.2f}'.format(mean_number_bids)) 
    median_number_bids = np.median(df_tenders['Number_bids'])
    print('Median value of bidders per tender: {:,.2f}'.format(median_number_bids)) 
    if 'Competitors' in df:
        df_competitors = df.drop_duplicates(subset=['Competitors'])
        number_competitors = len(df_competitors)
        print('Competitors: {0}'.format(number_competitors))
        number_winners = len(df_tenders['Competitors'].unique())
        print('Winning competitors of tenders: {} ({:,.2f}%)'.format(number_winners, number_winners/number_competitors*100))

    # Collusive vs competitive bids and tenders
    number_collusive_tenders = len(df_tenders[df_tenders['Collusive_competitor'] == 1])
    print('Collusive tenders: {} ({:,.2f}%)'.format(number_collusive_tenders, number_collusive_tenders/number_tenders*100))
    number_competitive_tenders = len(df_tenders[df_tenders['Collusive_competitor'] == 0])
    print('Competitive tenders: {} ({:,.2f}%)'.format(number_competitive_tenders, number_competitive_tenders/number_tenders*100)) 
    print('Collusive bids: {} ({:,.2f}%)'.format(number_collusive_bidders, number_collusive_bidders/number_bids*100))
    number_competitive_bidders = len(df[df[column_name] == 0])
    print('Competitive bids: {} ({:,.2f}%)'.format(number_competitive_bidders, number_competitive_bidders/number_bids*100))      
    if 'Competitors' in df:
        number_collusive_competitors = len(df_competitors[df_competitors['Collusive_competitor'] == 1])
        print('Collusive competitors: {} ({:,.2f}%)'.format(number_collusive_competitors, number_collusive_competitors/number_competitors*100))
        number_competitive_competitors = len(df_competitors[df_competitors['Collusive_competitor'] == 0])
        print('Competitive competitors: {} ({:,.2f}%)'.format(number_competitive_competitors, number_competitive_competitors/number_competitors*100)) 

    # Number of tenders by received offers: 1-4, 5-10, >10
    tenders_by_offer_group_1 = len(df_tenders[df_tenders['Number_bids'] <= 4])
    print('Bids by tender: 1<=N<=4: {} ({:,.2f}%)'.format(tenders_by_offer_group_1, tenders_by_offer_group_1/number_tenders*100))
    tenders_by_offers_group_2 = len(df_tenders[df_tenders['Number_bids'] <= 10]) - tenders_by_offer_group_1
    print('Bids by tender: 5<=N<=10: {} ({:,.2f}%)'.format(tenders_by_offers_group_2, tenders_by_offers_group_2/number_tenders*100))
    tenders_by_offers_group_3 = len(df_tenders[df_tenders['Number_bids'] > 10])
    print('Bids by tender: 11<=N: {} ({:,.2f}%)'.format(tenders_by_offers_group_3, tenders_by_offers_group_3/number_tenders*100))

    # Values of the winner's bid
    df_winners = df[df['Winner'] == 1]
    aggregated_bid_value = df_winners['Bid_value'].sum()
    print('Aggregated tender price: {:,.0f}'.format(aggregated_bid_value))
    aggregated_collusive_bid_value = df_winners[df_winners[column_name] == 1]['Bid_value'].sum()
    print('Aggregated collusive tender price: {:,.0f} ({:,.2f}%)'.format(aggregated_collusive_bid_value, aggregated_collusive_bid_value/aggregated_bid_value*100))
    aggregated_competitive_bid_value = df_winners[df_winners[column_name] == 0]['Bid_value'].sum()
    print('Aggregated competitive tender price: {:,.0f} ({:,.2f}%)'.format(aggregated_competitive_bid_value, aggregated_competitive_bid_value/aggregated_bid_value*100))
    mean_bid_value = np.mean(df_winners['Bid_value'])
    print('Mean tender price: {:,.2f}'.format(mean_bid_value))
    median_bid_value = np.median(df_winners['Bid_value'])
    print('Median tender price: {:,.2f}'.format(median_bid_value))
    print('------------------------------------')
    print('')


def get_dataset(dataset):
    ''' Get the collusive dataset and their fields to use in the ML algorimths '''

    predictors = defaultdict(list)

    if dataset == 'brazilian':
        df_collusion = pd.read_csv(db_collusion_brazilian, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Pre-Tender Estimate (PTE)', 'Difference Bid/PTE', 'Site', 'Date', 'Brazilian State', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Date', 'Number_bids']
    
    elif dataset == 'switzerland_gr_sg':
        df_collusion = pd.read_csv(db_collusion_switzerland_gr_sg, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Contract_type', 'Date', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Date', 'Number_bids']
    
    elif dataset == 'switzerland_ticino':
        df_collusion = pd.read_csv(db_collusion_switzerland_ticino, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Consortium', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Number_bids']
    
    elif dataset == 'italian':
        df_collusion = pd.read_csv(db_collusion_italian, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Pre-Tender Estimate (PTE)', 'Difference Bid/PTE', 'Site', 'Capital', 'Legal_entity_type', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Number_bids']
    
    elif dataset == 'american':
        df_collusion = pd.read_csv(db_collusion_american, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Bid_value_without_inflation', 'Bid_value_inflation_raw_milk_price_adjusted_bid', 'Date', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Date', 'Number_bids']
    
    elif dataset == 'japan':
        df_collusion = pd.read_csv(db_collusion_japan, header=0)
        predictors['all_setting'] = ['Tender', 'Bid_value', 'Pre-Tender Estimate (PTE)', 'Difference Bid/PTE', 'Site', 'Date', 'Winner', 'Number_bids']
        predictors['all_setting+screens'] = predictors['all_setting'] + screens
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Date', 'Number_bids']
    
    elif dataset == 'all':
        df_collusion = pd.read_csv(db_collusion_all, header=0)
        predictors['common'] = ['Tender', 'Bid_value', 'Winner', 'Number_bids', 'Dataset']

    predictors['common+screens'] = predictors['common'] + screens
    
    # Output fields of the datasets to the ML algorithms.
    targets = ['Collusive_competitor'] 

    return df_collusion, predictors, targets 


if __name__ == '__main__':
    
    # The user selectes the dataset to analyse
    dataset = None
    while dataset == None:
        number_input = input('Insert the number to analyse the dataset [brazilian (1), american (2), italian (3), ' \
                         'switzerland_gr_sg (4), switzerland_ticino (5), japan (6), all datasets (7) or exit (0)]: ')
        if number_input == '0': sys.exit(0)
        elif number_input == '1': dataset = 'brazilian'
        elif number_input == '2': dataset = 'american'
        elif number_input == '3': dataset = 'italian'
        elif number_input == '4': dataset = 'switzerland_gr_sg'
        elif number_input == '5': dataset = 'switzerland_ticino'
        elif number_input == '6': dataset = 'japan'       
        elif number_input == '7': dataset = 'all'     

    # 1. Get the dataset processed ready to use with the ML algorithms
    df_collusion, predictors, targets = get_dataset(dataset)
    
    # 2. Print information of the processed datasets
    print_description_processed_dataset(df_collusion)
    
    # 3. Print list with the colluded tenders by bidder
    #calculate_colluded_tenders_by_bidder(df_collusion)

    # 4. Print Scatter Matrix
    df_scatter_matrix = shuffle_tenders(df_collusion)
    # Columns to plot
    columns_to_plot = ['Bid_value'] + screens + ['Collusive_competitor'] # Collusive_competitor is deleted at the end
    df_scatter_matrix = df_scatter_matrix[columns_to_plot]
    # Replace labels for colors to print the scatter matrix
    if 'Collusive_competitor_original' in df_scatter_matrix:
        df_scatter_matrix['Collusive_competitor'] = df_scatter_matrix['Collusive_competitor_original']
    colors_legend = ['Green', 'Red']
    labels_legend = ['Competitive bid', 'Collusive bid']
    df_color_labels = df_scatter_matrix[['Collusive_competitor']].replace(0, colors_legend[0])
    df_color_labels = df_color_labels[['Collusive_competitor']].replace(1, colors_legend[1])
    list_color_labels = df_color_labels['Collusive_competitor'].values.tolist()
    df_scatter_matrix.drop(columns=['Collusive_competitor'], inplace=True)
    printScatterMatrix(df_scatter_matrix, list_color_labels, colors_legend, labels_legend, dataset)

    # 5. Boxplots of screen variables
    # Check with len of screens
    max_ylim_screens = [0.4, 1.8, 0.3, 12, 4, 4, 1]
    min_ylim_screens = [0, 0, 0, -12, -4, -4, 0]
    step_y_screens = [0.4/10, 1.8/10, 0.3/10, 24/10, 8/10, 8/10, 1/10]
    df_collusion_copy = df_collusion.copy()
    df_collusion_copy['Collusive_competitor'].replace(0, 'Comp.', inplace=True)
    df_collusion_copy['Collusive_competitor'].replace(1, 'Coll.', inplace=True)
    for index, screen_variable in enumerate(screens):
        print_boxplot(df_collusion_copy, dataset, column_names=screen_variable, groupby='Collusive_competitor', min_ylim=min_ylim_screens[index], 
                      max_ylim=max_ylim_screens[index], step_y=step_y_screens[index], xlabel='Bids', percentage=True)

    # 6. Histogram or density plot of Number of Bids by Tender. Each plot for collusive tenders and honest tenders.

    df_hist = df_collusion
    if 'Collusive_competitor_original' in df_hist:
        df_hist['Collusive_competitor'] = df_hist['Collusive_competitor_original']
    competitive_bids = df_hist[df_hist['Collusive_competitor'] == 0]['Number_bids']
    collusive_bids = df_hist[df_hist['Collusive_competitor'] != 0]['Number_bids']
    plotTwoHistograms(competitive_bids, collusive_bids, label_1='competitive bids', label_2='collusive bids', max_range=125, bins=25, max_xlim=125, density=True)  

    # 7. Execute algorithm comparison and print table comparison  
    algorithm_comparison(df_collusion, dataset, predictors, targets, ml_algorithms, train_size, repetitions, n_estimators, precision_recall, load_data, save_data)
   
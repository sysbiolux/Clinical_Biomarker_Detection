#########################################################
# Test to find out how often things need to be shuffled
#http://www.ams.org/publicoutreach/feature-column/fcarc-shuffle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

########################################################
# FIRST EXAMPLE
# Hypothesis: Shuffle a vector as many times as its length n to enable each patient the chance to get all the values
# of the other patients. For this we select the position of one patient to trace back, say index 0, and store the new
# position after each shuffle in a result vector. Count the number of unique positions after n shuffles and compare with
# the real vector length n. How often do you really need to shuffle this vector to enable that each patient gets the
# values of the other patients at least once.
#######################################################

# Let's say we have a set of 1200 clinical patients, ranging from 0 to 1199
vector_fix = np.arange(1200)
# we create a copy of the vector to be shuffles
vector_to_shuffle = vector_fix.copy()
# generate a random_state seed
random_state = 42
# initiate and reset random generator
rng = np.random.RandomState(random_state)
# how many times should the vector be shuffled
shuffle = 1200
# select one patient position to trace back during shuffling
position_to_trace = 50
# initialize list of traced positions (without the first position '0' in vector_fix as this position is non-shuffled)
point = []
# we create a short function to easily find the position of a given traced number in an array


def find(x, array):
    for n in range(len(array)):
        if array[n] == x:
            return n


# iterate through number of shuffles
for i in range(shuffle):
    # reset the array to its initial position
    X = vector_fix.copy()
    # shuffle the array
    rng.shuffle(X)
    # get the 'position_to_trace' index
    point.append(find(position_to_trace, X))
# plot the retraced positions of the patient of interest across the number of shuffling
fig1 = plt.subplots()
plt.scatter(range(1, shuffle+1), point, color='k', s=2)
plt.scatter(0, position_to_trace, color='r', s=2)
plt.title(f'Retraced positions of patient index {position_to_trace} after\n{shuffle} '
          f'times shuffling a vector of {len(vector_fix)} length', fontsize=18)
plt.xlabel('times shuffled', fontsize=14)
plt.ylabel(f'retraced position of {position_to_trace}', fontsize=14)
plt.show(block=False)


# let's try to animate this using an frame updated (takes long to plot and save...)
def update(frame_number):

    ax.scatter(frame_number, point[frame_number], color='r', s=5, marker='o')
    line = ax.axvline(frame_number, ymin=0, ymax=1200, ls='--', lw=1)
    plt.pause(0.0001)
    line.remove()
    return scatter,


# starting the animated plot
fig = plt.figure(figsize=(12, 7))
ax = plt.axes(xlim=(-10, shuffle+10), ylim=(-10, 1200+10))
plt.title(f'Retraced positions of patient index {position_to_trace} after\n{shuffle} '
          f'times shuffling a vector of {len(vector_fix)} length', fontsize=18)
plt.xlabel('times shuffled', fontsize=14)
plt.ylabel(f'retraced position of {position_to_trace}', fontsize=14)
scatter = ax.scatter(range(shuffle), point, color='k', s=5)
# launch animation
anim = animation.FuncAnimation(fig, update, frames=1200, interval=1, save_count=1200, repeat=False)
plt.show()
# finally save as gif
plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\jeff.didier\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
f = r"trace_position.gif"
writer = animation.FFMpegWriter(fps=60)
anim.save(f, writer)


# count the number of unique positions occupied by the patient
count = 1
for i in range(len(vector_fix)):
    if i in point:
        count += 1
print(f'The traced value {position_to_trace} was found on {count} of {len(vector_fix)} '
      f'spots after {shuffle} times shuffling.')

########################################################################################################################
# I notice that after len(samples) times of shuffling, the traced position (tested 0 and 50) does not appear as many
# times as the vector has been shuffled. Let's try to create a function to see if this can be observed for each number.


def shuffled_position(random_state=None, n_shuffles=1, vector_fix=None):
    # initialize random number generator for reproducibility
    rng = np.random.RandomState(random_state)
    # set up matrix of results with shape (n_shuffles, len(vector_fix))
    result = np.zeros(shape=(len(vector_fix), n_shuffles))
    # for loop going through each possible position an through each times of shuffling
    for pos_to_trace in range(len(vector_fix)):
        for times_shuffled in range(n_shuffles):
            # each shuffle is applied on the restored fixed vector
            array_to_shuffle = vector_fix.copy()
            # shuffle the copy
            rng.shuffle(array_to_shuffle)
            # store the traced position in the shuffled array into result matrix
            result[pos_to_trace, times_shuffled] = find(pos_to_trace, array_to_shuffle)
    # return the result matrix
    return result


# function to count the number of unique positions in the result matrix
def nunique(a, axis):
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis)+1


# function to get the plotting points
def get_plot_points(result=None, vector_fix=None):
    plot_points = []
    for k in range(len(vector_fix)):
        counts = nunique(result[k, :], axis=0)
        plot_points.append(counts)
    return plot_points


# get the shuffled position matrix as res (random_state, shuffle and vector_fix are referenced on top of script)
res = shuffled_position(random_state=random_state, n_shuffles=shuffle, vector_fix=vector_fix)  # takes some time (5 min)
# get the number of occupied position for each traced number in plot_points
plot_points = get_plot_points(result=res, vector_fix=vector_fix)

# plot the findings
fig2 = plt.subplots()
plt.scatter(range(len(vector_fix)), plot_points, color='k', s=2)
plt.title(f'Total occupied positions of each sample after shuffling n times\n'
          f'(n = length of input vector = {len(vector_fix)})')
plt.xlabel('Sample position')
plt.ylabel(f'Counted unique occupations after n shuffles')
plt.show(block=False)

# what is the mean occupation number
mean_occupation = np.mean(plot_points)
min_occupation = plot_points[np.argmin(plot_points)]
max_occupation = plot_points[np.argmax(plot_points)]

print(f'mean occupation: {mean_occupation}, max occupation: {max_occupation}, min occupation: {min_occupation}')

# Observations: mean occupation: 758.81, max occupation: 792, min occupation: 724 of 1200 samples
# equaling in percentages:       63.23%                  66%               60.33% of 1200 samples
# This means that shuffling a vector as many times as its size does equal to approx. 2/3 coverage
# How many times does it require to reach mean occupation equaling to the number of total samples
##################################################################################################

# empiric approach by trying different number of shuffle for a vector of 1200, say 10%, 25%, 50%, 75%, 100%, 125%, 150%)
n_shuffles = np.array([20, 50, 100, 150, 200,
                       250, 300, 350, 400, 450,
                       500, 550, 600, 650, 700, 800,
                       900, 1000, 1100, 1200, 1300,
                       1400, 1500, 1600, 1800, 2000,
                       2200, 2400, 2600, 2800, 3000,
                       3200, 3400, 3600, 3800, 4000])
vector_fix = np.arange(1200)

means = []
mins = []
maxs = []
# Following for loop might take 2-3 hours
for party in n_shuffles:
    output = shuffled_position(random_state=random_state, n_shuffles=int(party), vector_fix=vector_fix)
    output_plot_points = plot_points = get_plot_points(result=output, vector_fix=vector_fix)
    mean_occupations = np.mean(output_plot_points)
    min_occupations = plot_points[np.argmin(output_plot_points)]
    max_occupations = plot_points[np.argmax(output_plot_points)]
    print(f'shuffles: {party}, mean occupation: {mean_occupations}, max occupation: {max_occupations}, min occupation: {min_occupations}')
    means.append(mean_occupations)
    mins.append(min_occupations)
    maxs.append(max_occupations)

# OBSERVATIONS FOR A FIXED VECTOR OF 1200 ENTRIES SHUFFLED 36 DIFFERENT TIMES (20 - 4000):
####################################################################################################################
# shuffles: 20,     mean occupation: 19.856666666666666,   max occupation: 20,      min occupation: 18
# shuffles: 50,     mean occupation: 48.994166666666665,   max occupation: 50,      min occupation: 45
# shuffles: 100,    mean occupation: 95.97666666666667,    max occupation: 100,     min occupation: 89
# shuffles: 150,    mean occupation: 141.0525,             max occupation: 148,     min occupation: 132
# shuffles: 200,    mean occupation: 184.22083333333333,   max occupation: 194,     min occupation: 167
# shuffles: 250,    mean occupation: 225.7225,             max occupation: 239,     min occupation: 212
# shuffles: 300,    mean occupation: 265.2758333333333,    max occupation: 281,     min occupation: 251
# shuffles: 350,    mean occupation: 303.4375,             max occupation: 322,     min occupation: 283
# shuffles: 400,    mean occupation: 340.185,              max occupation: 360,     min occupation: 319
# shuffles: 450,    mean occupation: 375.37166666666667,   max occupation: 398,     min occupation: 354
# shuffles: 500,    mean occupation: 409.2341666666667,    max occupation: 431,     min occupation: 382
# shuffles: 550,    mean occupation: 441.2316666666667,    max occupation: 465,     min occupation: 416
# shuffles: 600,    mean occupation: 472.51,               max occupation: 501,     min occupation: 446
# shuffles: 650,    mean occupation: 502.1983333333333,    max occupation: 530,     min occupation: 469
# shuffles: 700,    mean occupation: 530.5141666666667,    max occupation: 560,     min occupation: 503
# shuffles: 800,    mean occupation: 583.6933333333334,    max occupation: 612,     min occupation: 553
# shuffles: 900,    mean occupation: 633.9191666666667,    max occupation: 663,     min occupation: 599
# shuffles: 1000,   mean occupation: 678.9166666666666,    max occupation: 718,     min occupation: 650
# shuffles: 1100,   mean occupation: 720.0808333333333,    max occupation: 756,     min occupation: 686
# shuffles: 1200,   mean occupation: 758.81,               max occupation: 792,     min occupation: 724
# shuffles: 1300,   mean occupation: 794.2708333333334,    max occupation: 833,     min occupation: 758
# shuffles: 1400,   mean occupation: 826.3283333333334,    max occupation: 868,     min occupation: 795
# shuffles: 1500,   mean occupation: 856.5341666666667,    max occupation: 889,     min occupation: 818
# shuffles: 1600,   mean occupation: 884.4166666666666,    max occupation: 928,     min occupation: 847
# shuffles: 1800,   mean occupation: 932.3383333333334,    max occupation: 974,     min occupation: 901
# shuffles: 2000,   mean occupation: 973.4508333333333,    max occupation: 1014,    min occupation: 942
# shuffles: 2200,   mean occupation: 1008.5466666666666,   max occupation: 1040,    min occupation: 980
# shuffles: 2400,   mean occupation: 1037.5466666666666,   max occupation: 1066,    min occupation: 1005
# shuffles: 2600,   mean occupation: 1063.3208333333334,   max occupation: 1097,    min occupation: 1028
# shuffles: 2800,   mean occupation: 1083.7483333333332,   max occupation: 1112,    min occupation: 1055
# shuffles: 3000,   mean occupation: 1101.8108333333332,   max occupation: 1135,    min occupation: 1070
# shuffles: 3200,   mean occupation: 1116.8966666666668,   max occupation: 1143,    min occupation: 1097
# shuffles: 3400,   mean occupation: 1129.635,             max occupation: 1153,    min occupation: 1104
# shuffles: 3600,   mean occupation: 1140.53,              max occupation: 1161,    min occupation: 1119
# shuffles: 3800,   mean occupation: 1149.7158333333334,   max occupation: 1168,    min occupation: 1127
# shuffles: 4000,   mean occupation: 1157.3516666666667,   max occupation: 1176,    min occupation: 1138
# ...
####################################################################################################################
# Stored values to save computational time:

means_computed = [19.856666666666666, 48.994166666666665, 95.97666666666667, 141.0525, 184.22083333333333, 225.7225,
                  265.2758333333333, 303.4375, 340.185, 375.37166666666667, 409.2341666666667, 441.2316666666667,
                  472.51, 502.1983333333333, 530.5141666666667, 583.6933333333334, 633.9191666666667, 678.9166666666666,
                  720.0808333333333, 758.81, 794.2708333333334, 826.3283333333334, 856.5341666666667, 884.4166666666666,
                  932.3383333333334, 973.4508333333333, 1008.5466666666666, 1037.5466666666666, 1063.3208333333334,
                  1083.7483333333332, 1101.8108333333332, 1116.8966666666668, 1129.635, 1140.53, 1149.7158333333334,
                  1157.3516666666667]
max_computed = [20, 50, 100, 148, 194, 239, 281, 322, 360, 398, 431, 465, 501, 530, 560, 612, 663, 718, 756, 792, 833,
                868, 889, 928, 974, 1014, 1040, 1066, 1097, 1112, 1135, 1143, 1153, 1161, 1168, 1176]

min_computed = [18, 45, 89, 132, 167, 212, 251, 283, 319, 354, 382, 416, 446, 469, 503, 553, 599, 650, 686, 724, 758,
                795, 818, 847, 901, 942, 980, 1005, 1028, 1055, 1070, 1097, 1104, 1119, 1127, 1138]

# plot the findings
fig3 = plt.subplots()
# about means
plt.scatter(n_shuffles, means_computed, color='k', s=5)
plt.hlines(y=len(vector_fix), xmin=0, xmax=n_shuffles[-1], linestyles='--', lw=1.5)
plt.hlines(y=len(vector_fix)/2, xmin=0, xmax=n_shuffles[-1], linestyles='-.', lw=1.5)
intersection = [g for g in range(len(means_computed)) if means_computed[g] >= len(vector_fix)/2][0]
plt.vlines(x=n_shuffles[intersection], ymin=0, ymax=len(vector_fix), linestyles=':', lw=1.5, color='k')
# about minima
plt.scatter(n_shuffles, min_computed, color='r', s=5, marker='^')
intersection_min = [g for g in range(len(min_computed)) if min_computed[g] >= len(vector_fix)/2][0]
plt.vlines(x=n_shuffles[intersection_min], ymin=0, ymax=len(vector_fix), linestyles=':', lw=1.5, color='r')
# add text
plt.text(x=n_shuffles[intersection] - 50, y=50, s=f'On average half of the positions\nare occupied after {n_shuffles[intersection]}\ntimes shuffling a vector of {len(vector_fix)}', ha='right')
plt.text(x=n_shuffles[intersection_min] + 50, y=50, s=f'At least half of the positions\nare occupied after {n_shuffles[intersection_min]}\ntimes shuffling a vector of {len(vector_fix)}')
plt.title(f'Total mean and minimum occupied positions of all samples after shuffling same vector increasing amount of times n\n', fontsize=18)
plt.xlabel('Numbers shuffled', fontsize=14)
plt.ylabel(f'Mean and minimum unique occupations after shuffling', fontsize=14)
plt.text(3200, -200, f'(n = {n_shuffles})', ha='center', fontsize=10)
plt.show(block=False)
####################################################################################################################
# Let's see what are the values for half of the data set (either male or female)

n_shuffles_small = np.array([20, 50, 100, 150, 200,
                       250, 300, 350, 400, 450,
                       500, 550, 600, 650, 700, 800,
                       900, 1000, 1100, 1200, 1300,
                       1400, 1500, 1600, 1800, 2000])
vector_fix_small = np.arange(600)

means_small = []
mins_small = []
maxs_small = []
# Following for loop might take 2-3 hours
for party in n_shuffles_small:
    output = shuffled_position(random_state=random_state, n_shuffles=int(party), vector_fix=vector_fix_small)
    output_plot_points = plot_points = get_plot_points(result=output, vector_fix=vector_fix_small)
    mean_occupations = np.mean(output_plot_points)
    min_occupations = plot_points[np.argmin(output_plot_points)]
    max_occupations = plot_points[np.argmax(output_plot_points)]
    print(f'shuffles: {party}, mean occupation: {mean_occupations}, max occupation: {max_occupations}, min occupation: {min_occupations}')
    means_small.append(mean_occupations)
    mins_small.append(min_occupations)
    maxs_small.append(max_occupations)

# OBSERVATIONS FOR A FIXED VECTOR OF 600 ENTRIES SHUFFLED 26 DIFFERENT TIMES (20 - 2000):
####################################################################################################################
# shuffles: 20,     mean occupation: 19.723333333333333,    max occupation: 20,     min occupation: 18
# shuffles: 50,     mean occupation: 47.906666666666666,    max occupation: 50,     min occupation: 42
# shuffles: 100,    mean occupation: 92.085,                max occupation: 100,    min occupation: 81
# shuffles: 150,    mean occupation: 132.87833333333333,    max occupation: 144,    min occupation: 124
# shuffles: 200,    mean occupation: 170.13333333333333,    max occupation: 184,    min occupation: 152
# shuffles: 250,    mean occupation: 204.8,                 max occupation: 217,    min occupation: 189
# shuffles: 300,    mean occupation: 236.20166666666665,    max occupation: 251,    min occupation: 220
# shuffles: 350,    mean occupation: 265.045,               max occupation: 283,    min occupation: 248
# shuffles: 400,    mean occupation: 292.2183333333333,     max occupation: 311,    min occupation: 272
# shuffles: 450,    mean occupation: 316.9316666666667,     max occupation: 338,    min occupation: 297
# shuffles: 500,    mean occupation: 339.66833333333335,    max occupation: 365,    min occupation: 319
# shuffles: 550,    mean occupation: 360.04,                max occupation: 382,    min occupation: 327
# shuffles: 600,    mean occupation: 379.3433333333333,     max occupation: 400,    min occupation: 354
# shuffles: 650,    mean occupation: 396.985,               max occupation: 418,    min occupation: 368
# shuffles: 700,    mean occupation: 413.505,               max occupation: 435,    min occupation: 390
# shuffles: 800,    mean occupation: 442.195,               max occupation: 477,    min occupation: 421
# shuffles: 900,    mean occupation: 466.37166666666667,    max occupation: 488,    min occupation: 443
# shuffles: 1000,   mean occupation: 486.8616666666667,     max occupation: 508,    min occupation: 467
# shuffles: 1100,   mean occupation: 504.16833333333335,    max occupation: 528,    min occupation: 483
# shuffles: 1200,   mean occupation: 519.0283333333333,     max occupation: 541,    min occupation: 499
# shuffles: 1300,   mean occupation: 530.95,                max occupation: 547,    min occupation: 507
# shuffles: 1400,   mean occupation: 541.915,               max occupation: 561,    min occupation: 518
# shuffles: 1500,   mean occupation: 550.4683333333334,     max occupation: 567,    min occupation: 533
# shuffles: 1600,   mean occupation: 558.3983333333333,     max occupation: 572,    min occupation: 540
# shuffles: 1800,   mean occupation: 570.3783333333333,     max occupation: 587,    min occupation: 550
# shuffles: 2000,   mean occupation: 578.5133333333333,     max occupation: 590,    min occupation: 565
# ...
####################################################################################################################
# Stored values to save computational time:

means_small_computed = [19.723333333333333, 47.906666666666666, 92.085, 132.87833333333333, 170.13333333333333,
                        204.8, 236.20166666666665, 265.045, 292.2183333333333, 316.9316666666667, 339.66833333333335,
                        360.04, 379.3433333333333, 396.985, 413.505, 442.195, 466.37166666666667, 486.8616666666667,
                        504.16833333333335, 519.0283333333333, 530.95, 541.915, 550.4683333333334, 558.3983333333333,
                        570.3783333333333, 578.5133333333333]
max_small_computed = [20, 50, 100, 144, 184, 217, 251, 283, 311, 338, 365, 382, 400, 418, 435, 477, 488, 508, 528, 541,
                      547, 561, 567, 572, 587, 590]
min_small_computed = [18, 42, 81, 124, 152, 189, 220, 248, 272, 297, 319, 327, 354, 368, 390, 421, 443, 467, 483, 499,
                      507, 518, 533, 540, 550, 565]

# plot the findings
fig4 = plt.subplots()
# about means
plt.scatter(n_shuffles_small, means_small_computed, color='k', s=5)
plt.hlines(y=len(vector_fix_small), xmin=0, xmax=n_shuffles_small[-1], linestyles='--', lw=1.5)
plt.hlines(y=len(vector_fix_small)/2, xmin=0, xmax=n_shuffles_small[-1], linestyles='-.', lw=1.5)
intersection = [p for p in range(len(means_small_computed)) if means_small_computed[p] >= len(vector_fix_small)/2][0]
plt.vlines(x=n_shuffles_small[intersection], ymin=0, ymax=len(vector_fix_small), linestyles=':', lw=1.5)
# about minima
plt.scatter(n_shuffles_small, min_small_computed, color='r', s=5, marker='^')
intersection_min = [p for p in range(len(min_small_computed)) if min_small_computed[p] >= len(vector_fix_small)/2][0]
plt.vlines(x=n_shuffles_small[intersection_min], ymin=0, ymax=len(vector_fix_small), linestyles=':', lw=1.5, color='r')
# add text
plt.text(x=n_shuffles_small[intersection] - 50, y=50, s=f'On average half of the positions\nare occupied after {n_shuffles_small[intersection]}\ntimes shuffling a vector of {len(vector_fix_small)}', ha='right')
plt.text(x=n_shuffles_small[intersection_min] + 50, y=50, s=f'At least half of the positions\nare occupied after {n_shuffles_small[intersection_min]}\ntimes shuffling a vector of {len(vector_fix_small)}')
plt.title(f'Total mean and minimum occupied positions of all samples after shuffling same vector increasing amount of times n\n', fontsize=18)
plt.xlabel('Numbers shuffled', fontsize=14)
plt.ylabel(f'Mean and minimum unique occupations after shuffling', fontsize=14)
plt.text(1600, -80, f'(n = {n_shuffles_small})', ha='center', fontsize=10)
plt.show(block=False)



# plot the findings
fig3 = plt.subplots()
# about means
plt.scatter(n_shuffles, means_computed, color='k', s=5)
plt.hlines(y=len(vector_fix), xmin=0, xmax=n_shuffles[-1], linestyles='--', lw=1.5)
plt.hlines(y=len(vector_fix)/2, xmin=0, xmax=n_shuffles[-1], linestyles='-.', lw=1.5)
intersection = [g for g in range(len(means_computed)) if means_computed[g] >= len(vector_fix)/2][0]
plt.vlines(x=n_shuffles[intersection], ymin=0, ymax=len(vector_fix), linestyles=':', lw=1.5, color='k')
# about minima
plt.scatter(n_shuffles, min_computed, color='r', s=5, marker='^')
intersection_min = [g for g in range(len(min_computed)) if min_computed[g] >= len(vector_fix)/2][0]
plt.vlines(x=n_shuffles[intersection_min], ymin=0, ymax=len(vector_fix), linestyles=':', lw=1.5, color='r')
# add text
plt.text(x=n_shuffles[intersection] - 50, y=50, s=f'On average half of the positions\nare occupied after {n_shuffles[intersection]}\ntimes shuffling a vector of {len(vector_fix)}', ha='right')
plt.text(x=n_shuffles[intersection_min] + 50, y=50, s=f'At least half of the positions\nare occupied after {n_shuffles[intersection_min]}\ntimes shuffling a vector of {len(vector_fix)}')
plt.title(f'Total mean and minimum occupied positions of all samples after shuffling same vector increasing amount of times n\n', fontsize=18)
plt.xlabel('Numbers shuffled', fontsize=14)
plt.ylabel(f'Mean and minimum unique occupations after shuffling', fontsize=14)
plt.text(3200, -200, f'(n = {n_shuffles})', ha='center', fontsize=10)
plt.show(block=False)
####################################################################################################################
# IN RESPECT TO THE MEAN
# OUTCOME: 900 shuffles for vector of 1200 if we want that on average each sample received
# at least half of the other values (e.g. in our full training set)
# OUTCOME: 450 shuffles for vector of 600 if we want that on average each sample received
# at least half of the other values (e.g. in our male/female training sets)
# IN RESPECT TO THE MIN
# OUTCOME: 1000 shuffles for vector of 1200 if we want that each sample received at least
# half of the other values (e.g. in our full training set)
# OUTCOME: 500 shuffles for vector of 600 if we want that each sample received at least
# half of the other values (e.g. in our male/female training sets)
###################################################################################################################

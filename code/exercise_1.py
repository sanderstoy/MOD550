# %%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os
import pandas as pd
import requests
    
class generate():
    ''' Initialization of the class '''
    def __init__(self, points, noise):
        self.points = points
        self.noise = np.random.randn(self.points)*noise
        self.x = np.linspace(0, 5, self.points)
        self.y = np.sin(np.pi*self.x)

    def generate_data(self):
        ''' Generate data with noise '''
        y = self.y * self.noise
        return y

    def plot(self):
        ''' Plot the data '''
        color = cm.spring(np.linspace(0, 1, self.points))
        plt.scatter(self.x, self.y, c=color)
        plt.title('Original plot before appending the two datasets', c='white')
        plt.xlabel('x', c='white')
        plt.ylabel('y', c='white')
        plt.gca().set_facecolor('black')
        plt.gcf().set_facecolor('black')
        plt.xticks(color='white')
        plt.yticks(color='white')
        ax = plt.gca()
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        plt.grid(color='white')
        plt.savefig('../data/plot_1.png', dpi=200)
        plt.show()

    def data2d(self):
        ''' Creates two datasets in 2d.
            Here 'zip' pairs each value of x with each value of y'''
        xy = list(zip(self.x, self.y))
        xy_noise_truth = list(zip(self.x, self.generate_data()))
        return xy, xy_noise_truth
    
    def append(self):
        ''' Append the data '''
        xy, xy_noise_truth = self.data2d()
        appended_2d = np.append(xy, xy_noise_truth, axis=0)
        return appended_2d
    
    def plot_new_data(self):
        ''' Plots the appended data.
            The appended data is a combination of the original data and the data with noise.
            First, the original data is plotted and then the original data with noise is plotted.'''
        appended_2d = self.append()
        color = cm.spring(np.linspace(0, 1, self.points))
        plt.scatter(appended_2d[:150,0], appended_2d[:150,1], c=color)
        plt.scatter(appended_2d[150:,0], appended_2d[150:,1], c=color)
        plt.title('Plot after appending the two datasets', color='white')
        plt.xlabel('x', c='white')
        plt.ylabel('y', c='white')
        plt.gca().set_facecolor('black')
        plt.gcf().set_facecolor('black')
        plt.xticks(color='white')
        plt.yticks(color='white')
        ax = plt.gca()
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        plt.grid(color='white')
        plt.savefig('../data/plot_2.png', dpi=200)
        plt.show()

    def save_dataset(self):
        ''' Converts the appended data into a dataframe 
            and then saves the dataframe as a .csv file '''
        filename = 'appended_data.csv'
        appended_2d = self.append()
        df = pd.DataFrame(appended_2d, columns=['x', 'y'])
        df.to_csv(f'../data/{filename}', index=False)

    def save_metadata(self):
        ''' Saves the metadata as a .json file '''
        appended_2d = self.append()
        metadata = {
                'plot': 'plot_2.png',
                'points': len(appended_2d),
                'noise': list(self.noise),
                'y-values': list(self.y)
            }
        with open('../data/metadata_plot_2.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        
    def create_txt_file(self):
        ''' Two lists are created, zipped and iterated over in order to write the content to a .txt file '''
        repositories = ['https://github.com/Mohamed-Hashem24/MOD550-Mohamed', 
                        'https://github.com/harry0703/MoneyPrinterTurbo', 
                        'https://github.com/stanford-oval/storm']
        
        comments = ['Sensible variable names has been used. The final class is well structured:', 
                    
                    'I have checked a few .py files and they seem to have used descriptive variable names, '
                    'but little to no comments in the code. '
                    'The README.md is written in both chinese and english - which is great for reaching a larger audience. '
                    'The folder structure is well organized:', 

                    'Many of the files includes helpful comments regarding class/functions functionality. '
                    'Filenames are descriptive and the folder structure is well organized. '
                    'A folder named "examples" is included, which is a great way to show how the code can be used.']
        
        repositories_comments = list(zip(repositories, comments))
        with open('../data/comment_on_repositories.txt', 'w') as f:
            for i in range(len(repositories_comments)):
                f.write('Repository:' +' '+ repositories_comments[i][0] + '\n' + 'Comment: ' + repositories_comments[i][1] + '\n\n')


    def integrate_new_dataset(self):
        ''' Fetches data and metadata from a github repository'''
        url_2d_data = 'https://raw.githubusercontent.com/Mohamed-Hashem24/Mohamed/refs/heads/main/MOD550/Text%20file/Appended_data_MH.csv'
        url_metadata = 'https://raw.githubusercontent.com/Mohamed-Hashem24/Mohamed/refs/heads/main/MOD550/Metadata/Mohamed_Hashem.json'
        integrated_data = pd.read_csv(url_2d_data)
        response = requests.get(url_metadata)
        integrated_metadata = response.json()
        integrated_metadata_df = pd.DataFrame([integrated_metadata])
        x = integrated_data['x']
        y = integrated_data['y']
        print(integrated_metadata_df)
        integrated_data.to_csv(f'../data/{'dataset_from_other_student.csv'}', index=False)
        return x, y
        # As the metadata only contain information about the file
        # and not the 2d data itself, I'll make a guess about the truth based on the .csv file
        # From the contents of the dataframe, it's clear that the dataset consists of 
        # seemingly random x-values between quite a big range (-100, 100), 
        # while the y-values range from 10-50. The x-values seem more frequent around +/- 0.
        # This is also confirmed by the plot. Therefore, my guess is that the student has
        # has created an x-array of random values centered around 0 and a y-array with values 
        # ranging from (10,50). Then, the x-array has been multiplied by an array of random values
        # between with a similar range as x but multiplied with a constant. This is the noise that
        # has been added to the dataset.


# Create an instance of the class
data = generate(150, 10)

# Call the functions
data.plot()
data.plot_new_data()
data.save_dataset()
data.save_metadata()
data.create_txt_file()
data.integrate_new_dataset()    


# My guess as to how the data has been generated alongside the imported data
x_imported, y_imported = data.integrate_new_dataset()
x = np.random.randn(500)
y = np.linspace(10, 50, 500)
noise = np.random.randn(500)*30
fig, ax = plt.subplots(1,2, figsize=(20,6))

ax[0].scatter(x_imported, y_imported)
ax[0].set_title('Imported dataset')
ax[1].scatter(x*noise,y, c='red')
ax[1].set_title('Assumed model')

for i in range(2):
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    ax[i].autoscale()
    ax[i].grid()

fig.savefig('../data/2d_plot_from_other_student_and_assumed_model.png', dpi=200)

# %%

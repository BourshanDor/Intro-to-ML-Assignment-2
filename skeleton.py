#################################
# Your name: Dor Bourshan
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
   



    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Implement me
        
        
        sample_D = np.ndarray(shape = (m,2))
        sample_D[:,0] = np.random.uniform(0,1,m)
        sample_D = sample_D[sample_D[:,0].argsort()] 
        for i in range(m) :
            x = sample_D[i][0]
            sample_D[i][1] = np.random.binomial(1, self.check_prob(x), 1)

        return sample_D




    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        x_axis = [] 
        y_true_axis = [] 
        y_empirical_axis = []

        for n in range(m_first, m_last+1, step) : 
            avg_true_error = 0 
            avg_empirical_error = 0 
            for i in range(T):
                sample = self.sample_from_D(n)
                interval, besterror = intervals.find_best_interval(sample[:,0], sample[:,1], k)
                avg_true_error += self.true_error(interval)
                avg_empirical_error += (besterror/n) 
            
            avg_true_error = avg_true_error/T 
            avg_empirical_error = avg_empirical_error/T 
            x_axis.append(n) 
            y_true_axis.append(avg_true_error)
            y_empirical_axis.append(avg_empirical_error )
        
        plt.plot(x_axis, y_true_axis, label="True Error")
        plt.plot(x_axis, y_empirical_axis, label="Empirical Error")
        plt.title("Empirical and True errors, averaged across the T runs")
        plt.xlabel("n")
        plt.ylabel("averaged across the T runs")
        plt.legend()
        plt.show()




    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        x_axis = [] 
        y_true_axis = [] 
        y_empirical_axis = []

        k_of_the_best_empirical_error =  0
        best_empirical_error = 1 
        sample = self.sample_from_D(m)
        sample_x = sample[:,0] 
        sample_lable = sample[:,1] 

        for k in range(k_first, k_last+1, step):
            interval, besterror = intervals.find_best_interval(sample_x, sample_lable, k)
            y_true_axis.append(self.true_error(interval))
            empirical_error = besterror/m
            y_empirical_axis.append(empirical_error) 
            x_axis.append(k)

            if best_empirical_error > empirical_error : 
                k_of_the_best_empirical_error = k
                best_empirical_error = empirical_error
            print(k)

        print("The best empirical k : %d" % k_of_the_best_empirical_error)
            
        plt.plot(x_axis, y_true_axis, label="True Error")
        plt.plot(x_axis, y_empirical_axis, label="Empirical Error")
        plt.title("empirical and true errors as a function of k")
        plt.xlabel("k")
        plt.ylabel("empirical and true errors ")
        plt.legend()
        plt.show()



    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        training_number = int(m*0.8) 
        test_number = m - training_number

        x_axis = [] 
        y_true_axis = [] 
        y_empirical_axis = []

        k_of_the_best_empirical_error =  0
        best_empirical_error = 1 
        train_sample = self.sample_from_D(training_number)
        test_sample = self.sample_from_D(test_number)
        train_sample_x = train_sample[:,0] 
        train_sample_lable = train_sample[:,1] 
    
        for k in range(1, 11):
            interval, besterror = intervals.find_best_interval(train_sample_x, train_sample_lable, k)
            y_true_axis.append(self.true_error(interval))
            empirical_error = self.empirical_error(interval, test_sample)
            y_empirical_axis.append(empirical_error) 
            x_axis.append(k)

            if best_empirical_error > empirical_error : 
                k_of_the_best_empirical_error = k
                best_empirical_error = empirical_error
            print(k)

        print("The best empirical k : %d" % k_of_the_best_empirical_error)
            
        plt.plot(x_axis, y_true_axis, label="True Error")
        plt.plot(x_axis, y_empirical_axis, label="Empirical Error")
        plt.title("empirical and true errors as a function of k")
        plt.xlabel("k")
        plt.ylabel("empirical and true errors ")
        plt.legend()
        plt.show()

    #################################
    # Place for additional methods


    #################################

    def check_prob(self,x) : 
        I = [np.array([0,0.2]), np.array([0.4,0.6]), np.array([0.8,1])]

        for interval in I : 
            if  self.belongs_to_interval(interval,x):  
                return .8 
        return .1

    def belongs_to_interval(self,interval, x) :
        return interval[0] <= x <= interval[1]   
    
    def complement_intervals(self,I) : 
        l = 0 
        r = I[0][0]
        I2 = [[l,r]] 
        for i in range(len(I)) :
            l = I[i][1]
            if i == len(I) -1 : 
                I2.append([l,1])
                break 
            r = I[i+1][0]
            I2.append([l,r])

        return I2

    
    def common_area(self, I1 , I2) : 
        """calculate the common area of I1 and I2""" 
        area = 0
        l = I1[0]
        r = I1[1] 
        for interval in I2 : 
            l2 = interval[0]
            r2 = interval[1]
            if l <= l2 and r >= r2 : 
                area += (r2 - l2)
            elif (l >= l2 and r <= r2) : 
                area += (r - l)
            elif (l2 <= r <= r2):
                area += (r - l2)
            elif (r2 >= l >= l2 ):
                area += (r2 - l)
        return area
        

    def true_error(self, I) :
        """e_p(h_I) = E[Z(h)] = E[h_I(X) ≠ Y] = Σ Pr[X ∈ [l_i,u_i] ] * Pr[h_I(X) ≠ Y | X ∈ [u_i, l_i] ]"""

        I2 = self.complement_intervals(I) 
        intervals_with_high_prob_label_1 = [[0,0.2],[0.4,0.6],[0.8,1]]  #0.8
        intervals_with_high_prob_label_0 = [[0.2,0.4],[0.6,0.8]]
        E = 0 

        for interval in I : #label '1'
            common_area_with_high_prob_area_to_secc_labeled  = self.common_area(interval, intervals_with_high_prob_label_1)
            common_area_with_low_prob_area_to_secc_labeled = (interval[1] - interval[0]) -   common_area_with_high_prob_area_to_secc_labeled
            assert common_area_with_low_prob_area_to_secc_labeled >= 0, "area should be non negative"
            E += 0.2*common_area_with_high_prob_area_to_secc_labeled + 0.9*common_area_with_low_prob_area_to_secc_labeled

        
        for interval in I2 : #label '0'
            common_area_with_high_prob_area_to_secc_labeled  = self.common_area(interval, intervals_with_high_prob_label_0)
            common_area_with_low_prob_area_to_secc_labeled = (interval[1] - interval[0]) -   common_area_with_high_prob_area_to_secc_labeled
            assert common_area_with_low_prob_area_to_secc_labeled >= 0, "area should be non negative"
            E += 0.1*common_area_with_high_prob_area_to_secc_labeled + 0.8*common_area_with_low_prob_area_to_secc_labeled

        return E
    

    def empirical_error(self, I , samples) :

        number_of_mistake = 0
        belong_to_interval = False 
        for sample in samples :
            x = sample[0]
            label = sample[1] 
            for interval in I : 
                if self.belongs_to_interval(interval, x) : 
                    belong_to_interval = True
                    if not( int(label) == 1 )  : # The label in the interval define as '1' 
                        number_of_mistake += 1   
                    break 
            if not( int(label) == 0) and not (belong_to_interval) :  # The label outside of the intervals define as '0' 
                number_of_mistake += 1 
        
        return number_of_mistake / np.size(samples, 0)
    
                
if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)


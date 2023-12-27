# baysian decoder adapted from maxcollard

class gNBPoissonClassifier:
    """A classifier that uses the naïve Bayes assumption with Poisson variates"""
    
    def __init__( self ):
        # Instance variables
        self.fit_ = False
        self.means_ = None
        self.prior_ = None
        self.n_classes_ = None
        self.n_neurons_ = None
        
    def fit( self, X, y ):
        """Fit the model
        
        Parameters
        ----------
        `X` - Input features (rows are trials, columns are features)
        `y` - Output classes (integers)
        """
        # Determine the shape from the data
        self.n_classes_ = np.nanmax( y ) + 1
        self.n_neurons_ = X.shape[1]
        n_trials = X.shape[0]
        
        # Fit prior using the empirical prevalence of the classes
        self.prior_ = np.zeros( (self.n_classes_,) )
        for c in range( self.n_classes_ ):
            # The empirical prior is the fraction of trials that have this class
            self.prior_[c] = np.nansum( y == c ) / n_trials
            # TRY ADDING IN UNIFORM PRIORS
            #self.prior_[c] = 1/self.n_classes_

        # Fit the class means using the empirical means
        self.means_ = np.zeros( (self.n_neurons_, self.n_classes_) )
        for c in range( self.n_classes_ ):
            # The empirical mean is the mean of all the trials that have this class
            idx_c = (y == c)
            self.means_[:, c] = np.nanmean( X[idx_c, :], axis = 0 )
        
        # Fit the class sd using the empirical means
        self.std_ = np.zeros( (self.n_neurons_, self.n_classes_) )
        for c in range( self.n_classes_ ):
            # The empirical mean is the mean of all the trials that have this class
            idx_c = (y == c)
            self.std_[:, c] = np.nanstd( X[idx_c, :], axis = 0 )
        

        # Set our fit flag
        self.fit_ = True
        
    def class_score_( self, c, xs ):
        """Assign the Bayes' rule output score for feature vector `xs`
        assuming the given class `c`"""
        # TODO This is the slowest possible way to do this
        ret = self.prior_[c]
        # Multiply the prior by each class' pmf
        for i, x in enumerate( xs ):
            #ret *= stats.poisson.pmf( x, self.means_[i, c] )
            ret *= stats.norm(self.means_[i, c]).pdf(x) #, self.std_[i, c]#gaussian way of pmf
            #the max before multipying ret is 0.3989422804014327
#         print( ret )
        return ret
        
    def class_scores_( self, xs ):
        # """Perform `self.class_score_` on the feature vector `xs` for each class"""
        # return np.array( [ self.class_score_( c, xs )
        #                    for c in range( self.n_classes_ ) ] )
        """Perform `self.class_score_` on the feature vector `xs` for each class"""
        scores = np.zeros(self.n_classes_)  # Initialize an array to store scores
        for c in range( self.n_classes_ ):
            scores[c] = self.class_score_(c, xs)
        return scores

    def best_class_( self, xs ):
        """Pick the best class given the Bayes' rule class scores for feature vector `xs`"""
        scores = self.class_scores_( xs )

        return np.nanargmax( scores )
        
    def predict( self, X, verbose = False ):
        """Predict the classes for the data in X
        
        Parameters
        ----------
        `X` - Input features (rows are trials, columns are features)
        `verbose` - Set to `True` for a progress bar (default: `False`)
        """
        
        if not self.fit_:
            raise ValueError( 'Classifier not fit yet!' )
        
        # Determine shape from the data
        n_trials = X.shape[0]
        
        # `it` is for "iterator"
        it = range( n_trials )
        if verbose:
            # If we set the verbose flag, give us a progress bar
            it = tqdm( it )
        
        # The output is the best class for each trial
        ret = np.array( [ self.best_class_( X[i_trial, :] )
                          for i_trial in it ] ).astype( int )
        return ret


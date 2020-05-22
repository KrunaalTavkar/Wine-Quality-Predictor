# Wine-Quality-Predictor
An implementation of the Linear Regression model designed from scratch, to predict the quality of white wine

# Description
Designed the Linear Regression Machine Learning model (from the ground up, without using existing libraries) and implemented the model on the White Wine Dataset from the UCI Machine Learning Repository. The model incorporates the handling of non-invertible matricies, an implementation of regularized Linear Regression to prevent overfitting and also tunes HyperParamters for Regularized Linear Regression.

The Performance Metric used is the Mean Absolute Error (MAE), which is defined as:

    𝑀𝐴𝐸=1𝑛∑𝑖=1𝑛|𝑦′−𝑦|
    
There is a further implementation of Polynomial Regression, which maps the given data to higher dimensions, as an experiment to see the correlation of the power (Dimension) and how the MAE changes with a change in power.

# Model Performance

    *** Simple Linear Regression ***
    Mapping and Power: False 0
    dimensionality of the model parameter is (12,).
    model parameter is  [ 2.03721116e+02  1.09955585e-01 -1.93164831e+00 -4.90845226e-02
                          1.02194195e-01 -5.45232538e-02  4.00189586e-03  1.52537227e-04
                         -2.04673020e+02  9.04608186e-01  6.41578532e-01  1.32320100e-01]
    MAE on train is 0.58020
    MAE on val is 0.59410
    MAE on test is 0.56079
    
    *** Handling Non-Invertible Matricies ***
    Mapping and Power: False 0
    dimensionality of the model parameter is (12,).
    model parameter is  [ 1.72490981 -0.04898016 -2.05742338 -0.12068529  0.02838304 -0.81308859
                          0.00425276  0.          0.00675103  0.20653332  0.35756603  0.37653797]
    MAE on train is 0.58509
    MAE on val is 0.59939
    MAE on test is 0.55777
    
    *** Regularized Linear Regression ***
    Mapping and Power: False 0
    dimensionality of the model parameter is (12,).
    model parameter is  [ 1.72490981 -0.04898016 -2.05742338 -0.12068529  0.02838304 -0.81308859
                          0.00425276  0.          0.00675103  0.20653332  0.35756603  0.37653797]
    MAE on train is 0.58509
    MAE on val is 0.59939
    MAE on test is 0.55777
    
    *** HyperParameter Tuning ***
    Mapping and Power: False 0
    BM: (3519, 12)
    Best Lambda =  1e-19
    dimensionality of the model parameter is 12.
    model parameter is  [ 2.03721116e+02  1.09955585e-01 -1.93164831e+00 -4.90845226e-02
                          1.02194195e-01 -5.45232538e-02  4.00189586e-03  1.52537227e-04
                         -2.04673020e+02  9.04608186e-01  6.41578532e-01  1.32320100e-01]
    MAE on train is 0.58020
    MAE on val is 0.59410
    MAE on test is 0.56079
    
    *** Polynomial Regression ***
    Power = 2
    MX DiM: (4399, 22)
    White Dim: (4399, 23)
    AM: (3519, 23)
    Best Lambda =  1e-05
    dimensionality of the model parameter is 23.
    model parameter is  [ 1.25380814e+02  5.02384266e-01 -2.67773310e+00  9.80340263e-01
                          1.06212461e-01 -3.95578940e+00  2.39784392e-02  7.50076068e-03
                         -3.12000991e+01 -4.86978603e+00  4.09420291e-01 -8.46119515e-01
                         -2.72604969e-02  1.29686830e+00 -1.25860412e+00 -8.91481755e-04
                          1.57808283e+01 -2.28910275e-04 -2.75880957e-05 -8.24960635e+01
                          8.96138511e-01  1.80781412e-01  4.41502041e-02]
    MAE on train is 0.56777
    MAE on val is 0.57459
    MAE on test is 0.57865
    
    Power = 3
    MX DiM: (4399, 33)
    White Dim: (4399, 34)
    (3519, 34)
    best lambd is 0.0001
    MAE on train is 0.56318
    MAE on val is 0.58117
    MAE on test is 0.58448

    Power = 4
    MX DiM: (4399, 44)
    White Dim: (4399, 45)
    (3519, 45)
    best lambd is 0.0001
    MAE on train is 0.56024
    MAE on val is 0.58280
    MAE on test is 1.27799

    Power = 5
    MX DiM: (4399, 55)
    White Dim: (4399, 56)
    (3519, 56)
    best lambd is 0.01
    MAE on train is 0.55637
    MAE on val is 0.57013
    MAE on test is 4.99187

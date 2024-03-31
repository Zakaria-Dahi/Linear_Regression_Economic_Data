
**Programmer:** Zakaria Abdelmoiz DAHI :shipit:

## **About the Repository:**

- This repository provides a linear regression model applied on economic data.
- The data represent 33 type of economic information related to 52 Spanish cities and recorded over the period of 15 years (2003-2017)

**About the Data:** :information_source:

- The data are real-life and extracted using the API of the Spanish national insitute of statistcs (https://www.ine.es/).
- The data have been recorded monthly during 15 years from  2003 to 2017.
- The data has been used  before in our work "**A Machine Learning-Based Approach for Economics-Tailored Applications: The Spanish Case Study**"

## **About the Metrics:** :straight_ruler:
- Four regression metrics are provide: Mean Absolute Error (MAE), Mean Square Error (MSE), Root Mean Square Error (RMSE), and R sqaure.

**How to use Regressor:** :notebook_with_decorative_cover:  

- The default execution of the regression is to repdict the Men Activity (i.e. employement) for the city of Ceuta during the year 2003.
- You can choose any city you want among the 52 Spanish cities, and also any of the ecnomic metrics among the 33 available ones, and finally you can pick the year you want to predict from 2003 to 2017.


## **The Regression You can Make:**
You can perform regression by setting the variables city, serie and year in the file ```main.py``` using one of the values indicated below for each of these variables.

  - **The Frameworks You Can Pick**
 
    - You can choose either linear regression using ```SciekitLearn``` or ```TensorFlow```.


  - **The Years You Can Pick**
 
You can choose any year from 2003 to 2017.

  - **The Cities You Can Pick:**

    - A Coruña, 
    Albacete, 
    Alicante, 
    Almería,
    Álava,
    Asturias,
    Ávila,
    Badajoz,
    Barcelona,
    Vizcaya,
    Burgos,
    Cáceres,
    Cádiz,
    Cantabria,
    Castellón,
    Ceuta,
    Ciudad Real,
    Córdoba,
    Cuenca,
    Guipúzcoa,
    Girona,
    Granada,
    Guadalajara,
    Huelva,
    Huesca,
    Baleares,
    Jaén,
    La Rioja,
    Las Palmas,
    León,
    Lleida,
    Lugo,
    Madrid,
    Málaga,
    Melilla,
    Murcia,
    Navarra,
    Ourense,
    Palencia,
    Pontevedra,
    Salamanca,
    Santa Cruz de Tenerife,
    Segovia,
    Sevilla,
    Soria,
    Tarragona,
    Teruel,
    Toledo,
    Valencia,
    Valladolid,
    Zamora,
    Zaragoza,

  - **The Metrics You Can Choose:**

    - Men Activity Percentage,
    Women Activity Percentage,
    Men Unemployment Percentage,
    Women Unemployment  Percentage,
    Men Employment Percentage,
    Women Employment  Percentage,
    Women Unemployment Percentage,
    Men employment Percentage,
    Women employment Percentage,
    Índice general. Variación mensual.,
    Alimentos y bebidas no alcohólicas. Índice.,
    Bebidas alcohólicas y tabaco. Índice.,
    Vestido y calzado. Índice.,
    Sanidad. Índice.,
    Transporte. Índice.,
    Comunicaciones. Índice.,
    Ocio y cultura. Índice.,
    Enseñanza. Índice.,
    Restaurantes y hoteles. Índice.,
    Otros bienes y servicios. Índice.,
    Sin asalariados. Total de empresas. Total CNAE. Empresas.,
    De 1 a 2. Total de empresas. Total CNAE. Empresas.,
    De 3 a 5. Total de empresas. Total CNAE. Empresas.,
    De 6 a 9. Total de empresas. Total CNAE. Empresas.,
    De 10 a 19. Total de empresas. Total CNAE. Empresas.,
    De 20 a 49. Total de empresas. Total CNAE. Empresas.,
    De 50 a 99. Total de empresas. Total CNAE. Empresas.,
    De 100 a 199. Total de empresas. Total CNAE. Empresas.,
    De 200 a 499. Total de empresas. Total CNAE. Empresas.,
    De 500 a 999. Total de empresas. Total CNAE. Empresas.,
    De 1000 a 4999. Total de empresas. Total CNAE. Empresas.,
    De 5000 o más asalariados. Total de empresas. Total CNAE. Empresas.,
    Total. Total de empresas. Total CNAE. Empresas.,

  - **About the Directories:** :open_file_folder:
    - The folder ```Input``` contains the dataset, while the folder ```Output``` contains two subfolders ```SL``` and ```FL```. Each of these two sufolders contains the numerical and graphical results of the training and testing of the linear regression model using ```SciekitLearn``` or ```TensorFlow```, respectively. 
    
## **Demo**
![tf_linear_prediction_training_dataCeuta_Men Activity Percentage_2003](https://user-images.githubusercontent.com/68249696/222951306-9f180ee7-fbf2-4d1e-b486-0ae3e9e2a778.png)
![tf_linear_prediction_testing_dataCeuta_Men Activity Percentage_2003](https://user-images.githubusercontent.com/68249696/222951310-4e050411-6cc0-4af3-aab9-986c12080601.png)


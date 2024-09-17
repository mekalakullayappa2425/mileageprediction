# mileageprediction
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPHT2G9MVR9rS/fy+KQPmM2",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/tmaniy/Mileage-Prediction-Model-Machine-Learning-/blob/main/Mileage_Prediction.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        " **MILEAGE PREDICTION MODEL : REGRESSION ANALYSIS**"
      ],
      "metadata": {
        "id": "2FmNXjLxfRkT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DATA SOURCE**:The dataset was taken from StateLib library which is maintained at Carnegie Mellon University"
      ],
      "metadata": {
        "id": "YnU57uNofRZN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**IMPORT LIBRARY**"
      ],
      "metadata": {
        "id": "CeuWjmMHfMTb"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XOKsHz6YPATN"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np"
      ],
      "metadata": {
        "id": "-MLhSINnPGxn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "_nBMdg5FPKb-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns"
      ],
      "metadata": {
        "id": "MOmrjqCDPTtG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**IMPORT DATA**"
      ],
      "metadata": {
        "id": "Q-IVNq7DfJKZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')"
      ],
      "metadata": {
        "id": "RW8SayXHQzg4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "id": "n9hKPCZ7Q0Dg",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "b2735dbf-96a7-4c18-ba5c-2f735e133c4a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": 
             ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DATA VISUALISATION**"
      ],
      "metadata": {
        "id": "BvDG6xrqe0QY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.pairplot(df, x_vars = ['mpg', 'displacement', 'horsepower', 'weight','acceleration'] , y_vars=['mpg'])"
      ],
      "metadata": {
        "id": "vkV9SVPaR74c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sns.regplot(x='displacement' , y='mpg', data = df)"
      ],
      "metadata": {
        "id": "iafY-yOjTLvN",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "outputId": "68f27a27-18ca-4dcd-cee4-9608546deb1e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='displacement', ylabel='mpg'>"
            ]
          },
          "metadata": {},
          "execution_count": 82
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
 ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DATA VISUALISATION**"
      ],
      "metadata": {
        "id": "BvDG6xrqe0QY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.pairplot(df, x_vars = ['mpg', 'displacement', 'horsepower', 'weight','acceleration'] , y_vars=['mpg'])"
      ],
      "metadata": {
        "id": "vkV9SVPaR74c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sns.regplot(x='displacement' , y='mpg', data = df)"
      ],
      "metadata": {
        "id": "iafY-yOjTLvN",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "outputId": "68f27a27-18ca-4dcd-cee4-9608546deb1e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='displacement', ylabel='mpg'>"
            ]
          },
          "metadata": {},
          "execution_count": 82
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
                    },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DEFINE TARGET VARIABLE Y AND FEATURE X**"
      ],
      "metadata": {
        "id": "TuclppRwejWX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns"
      ],
      "metadata": {
        "id": "sje4MOq1TZOF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ba6d609f-3bc2-4958-ec80-3c55a41399d6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',\n",
              "       'acceleration', 'model_year', 'origin', 'name'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y=df['mpg']"
      ],
      "metadata": {
        "id": "WA-nl9f1Tg03"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y.shape"
      ],
      "metadata": {
        "id": "b75Y2W3KTkn2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x=df[['displacement', 'horsepower', 'weight','acceleration']]"
      ],
      "metadata": {
        "id": "knjvylB-Trgz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x.shape"
      ],
      "metadata": {
        "id": "-jL5i6oDT11R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x"
      ],
      "metadata": {
        "id": "MzJTmgbfT4Ma",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "outputId": "780010e3-a3f9-4035-fccd-8147f750b7ad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
                  "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 74
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler"
      ],
      "metadata": {
        "id": "nQL1_uANUDjM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ss= StandardScaler()"
      ],
      "metadata": {
        "id": "3B0bnVCxUOFs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x = ss.fit_transform(x)"
      ],
      "metadata": {
        "id": "KQg3StdCUUmL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x"
      ],
      "metadata": {
        "id": "3_phKttmUgt2",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "outputId": "4cee4b58-cec1-4c03-df76-ac7580cfd8bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
                 ]
          },
          "metadata": {},
          "execution_count": 85
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(x).describe()"
      ],
      "metadata": {
        "id": "1pBoHsJ1UjQ8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "2eb4ffa3-f799-4740-8325-63a26a44f891"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
                ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "**SPLIT DATA INTO TRAIN TEST**"
      ],
      "metadata": {
        "id": "XOS1ytG3eCKS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "x0HuoBshUu1W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7 , random_state=2529)"
      ],
      "metadata": {
        "id": "_V01jGguVHqS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train.shape, x_test.shape, y_train.shape, y_test.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tdNyZAckVIJl",
        "outputId": "1ba870b7-2a4f-46a3-e08c-d21cc00fb243"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((274, 4), (118, 4), (274,), (118,))"
            ]
          },
          "metadata": {},
          "execution_count": 87
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**LINEAR REGRESSION MODEL**"
      ],
      "metadata": {
        "id": "A7UXg_RUd8Tv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression"
      ],
      "metadata": {
        "id": "wByO9N3YVIiI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr=LinearRegression()"
      ],
      "metadata": {
        "id": "-8f1rc8LVI5f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr.fit(x_train,y_train)"
      ],
      "metadata": {
        "id": "yMlu2R3uW5Ku",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "681704fa-4d77-4117-bbb1-e83d20ad2ce7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr.intercept_"
      ],
      "metadata": {
        "id": "2lpccV8RW5wi",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b9a9e737-4087-4188-bc6e-abb90bb70e42"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "23.485738559737584"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr.coef_"
      ],
      "metadata": {
        "id": "2CNSkUj2XdIM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "83a07065-a584-4090-b71e-95783cb1f8a4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-1.05767743, -1.68734727, -4.10787617, -0.11495177])"
            ]
          },
          "metadata": {},
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Mileage = 23.4-1.05(Displacement)-1.68(Horsepower)-4.10(weight)-0.115(Acceleration)+Error**"
      ],
      "metadata": {
        "id": "88Vl6JyxdbBu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**PREDICT TEST DATA**"
      ],
      "metadata": {
        "id": "FqdZCMgVdJx0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred=lr.predict(x_test)"
      ],
      "metadata": {
        "id": "EPbGfnrjXhKr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred"
      ],
      "metadata": {
        "id": "6aC6b_PFXqgb",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "179e56d8-6ee6-4b1d-ad82-eee062ccc15d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
                  ]
          },
          "metadata": {},
          "execution_count": 92
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**MODEL ACCURACY**"
      ],
      "metadata": {
        "id": "Tf75zCGdc33O"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score"
      ],
      "metadata": {
        "id": "YZt9p1HYXt32"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test,y_pred)"
      ],
      "metadata": {
        "id": "apTYzCQgYkup",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "531ae718-56a9-48c9-8b14-aa38801f1653"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3.3286968643244106"
            ]
          },
          "metadata": {},
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_percentage_error(y_test,y_pred)"
      ],
      "metadata": {
        "id": "rDcrtNsEYa6P",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "45ae41a1-5d6a-4960-f0e2-31d8c5d42fcb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.14713035779536746"
            ]
          },
          "metadata": {},
          "execution_count": 94
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test,y_pred)"
      ],
      "metadata": {
        "id": "-_UehLRvYbON",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "566a3451-62e1-4afd-c331-139e36ed0e08"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7031250746717691"
            ]
          },
          "metadata": {},
          "execution_count": 95
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**POLYNOMIAL REGRESSION**"
      ],
      "metadata": {
        "id": "D1hwHlrmcw4M"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import PolynomialFeatures"
      ],
      "metadata": {
        "id": "TOJCANXAYb3Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "poly=PolynomialFeatures(degree=2,interaction_only=True, include_bias=False)"
      ],
      "metadata": {
        "id": "l59CdwBnZFTZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train2 = poly.fit_transform(x_train)"
      ],
      "metadata": {
        "id": "FN0WikIKZF7P"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_test2 = poly.fit_transform(x_test)"
      ],
      "metadata": {
        "id": "-s7n29DbZGWL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr.fit(X_train2,y_train)"
      ],
      "metadata": {
        "id": "kYyQ_T0XZ71s",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "c8add60b-1aa2-48af-d590-8f316c2caaa4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 110
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr.intercept_"
      ],
      "metadata": {
        "id": "nAHksT4gaFTy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr.coef_"
      ],
      "metadata": {
        "id": "6G8Psc3TaFok"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred_poly = lr.predict(X_test2)"
      ],
      "metadata": {
        "id": "l7YRh5OuaMVL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**MODEL ACCURACY**"
      ],
      "metadata": {
        "id": "7JfQq6R5chT_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score"
      ],
      "metadata": {
        "id": "d0ZROfssaacP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test,y_pred_poly)"
      ],
      "metadata": {
        "id": "sqzw4_-NcNHZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7cc7328d-32f0-41e2-871b-6f218be0ff9a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2.7887147720295977"
            ]
          },
          "metadata": {},
          "execution_count": 105
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_percentage_error(y_test,y_pred_poly)"
      ],
      "metadata": {
        "id": "r6wVmyBkcVHd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5945f271-01e9-41b4-95cc-a5535109c1e8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.12074018342938687"
            ]
          },
          "metadata": {},
          "execution_count": 106
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test,y_pred_poly)"
      ],
      "metadata": {
        "id": "wsQ2huZwcVjX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f8dcac4f-d902-4ba0-b586-a5b13d3b11c9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7461731314563803"
            ]
          },
          "metadata": {},
          "execution_count": 107
        }
      ]
    }
  ]
}

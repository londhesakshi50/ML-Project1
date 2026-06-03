{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MpwqBD1RuYbJ"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wtS_vbUnw6PU"
      },
      "source": [
        "1.Problem Statement:"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EhfiPre5xYwM"
      },
      "source": [
        "Developed a machine learning model to estimate health insurance premiums based on customer attributes."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DGOr75WkyRDu"
      },
      "source": [
        "Independent variables:\n",
        "1. Age\n",
        "2. Gender\n",
        "3. Hieght\n",
        "4. weight\n",
        "5. Health Insurance Cover\n",
        "\n",
        "    \n",
        "\n",
        "\n",
        "Dependent Variable:\n",
        "\n",
        "Premium\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mPI2VH201uKL"
      },
      "source": [
        "2. Data Gathring:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "dYEu2Rte4nKO",
        "outputId": "ad431e50-09a3-4fdb-8eea-00a65c2ae928"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1676,\n  \"fields\": [\n    {\n      \"column\": \"Sr.no\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 483,\n        \"min\": 1,\n        \"max\": 1676,\n        \"num_unique_values\": 1676,\n        \"samples\": [\n          865,\n          1269,\n          417\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 353,\n        \"samples\": [\n          \"Angel\",\n          \"akshay\",\n          \"Nathan\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age(yrs)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 26.338582349667547,\n        \"min\": 0.25,\n        \"max\": 85.0,\n        \"num_unique_values\": 94,\n        \"samples\": [\n          34.0,\n          16.0,\n          49.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Height (cm)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 31.425153146843996,\n        \"min\": 56.0,\n        \"max\": 177.0,\n        \"num_unique_values\": 161,\n        \"samples\": [\n          113.5,\n          67.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Weight(kg)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 22.966761832996312,\n        \"min\": 4.8,\n        \"max\": 88.0,\n        \"num_unique_values\": 274,\n        \"samples\": [\n          70.0,\n          79.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Health Insurance cover\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3170480,\n        \"min\": 500000,\n        \"max\": 10000000,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          7500000,\n          750000\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Premium\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 25308,\n        \"min\": 7015,\n        \"max\": 111340,\n        \"num_unique_values\": 82,\n        \"samples\": [\n          26410,\n          7015\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "df"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-81d75e72-1af2-46a1-9b8e-88c26fc8f2f7\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sr.no</th>\n",
              "      <th>Name</th>\n",
              "      <th>Age(yrs)</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Height (cm)</th>\n",
              "      <th>Weight(kg)</th>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <th>Premium</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>krushna</td>\n",
              "      <td>0.250000</td>\n",
              "      <td>male</td>\n",
              "      <td>61.4</td>\n",
              "      <td>6.4</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>pavan</td>\n",
              "      <td>0.333333</td>\n",
              "      <td>male</td>\n",
              "      <td>63.9</td>\n",
              "      <td>7.0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>ketan</td>\n",
              "      <td>0.416667</td>\n",
              "      <td>male</td>\n",
              "      <td>65.9</td>\n",
              "      <td>7.5</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>akash</td>\n",
              "      <td>0.500000</td>\n",
              "      <td>male</td>\n",
              "      <td>67.6</td>\n",
              "      <td>7.9</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>syam</td>\n",
              "      <td>0.583333</td>\n",
              "      <td>male</td>\n",
              "      <td>69.2</td>\n",
              "      <td>8.3</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1671</th>\n",
              "      <td>1672</td>\n",
              "      <td>kiran</td>\n",
              "      <td>81.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1672</th>\n",
              "      <td>1673</td>\n",
              "      <td>mira</td>\n",
              "      <td>82.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1673</th>\n",
              "      <td>1674</td>\n",
              "      <td>radha</td>\n",
              "      <td>83.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1674</th>\n",
              "      <td>1675</td>\n",
              "      <td>lakshmi</td>\n",
              "      <td>84.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1675</th>\n",
              "      <td>1676</td>\n",
              "      <td>damini</td>\n",
              "      <td>85.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1676 rows × 8 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-81d75e72-1af2-46a1-9b8e-88c26fc8f2f7')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-81d75e72-1af2-46a1-9b8e-88c26fc8f2f7 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-81d75e72-1af2-46a1-9b8e-88c26fc8f2f7');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "  <div id=\"id_4d2dedec-6d86-4603-8150-a201bee7b65d\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('df')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_4d2dedec-6d86-4603-8150-a201bee7b65d button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('df');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "      Sr.no     Name   Age(yrs)  Gender  Height (cm)  Weight(kg)  \\\n",
              "0         1  krushna   0.250000    male         61.4         6.4   \n",
              "1         2    pavan   0.333333    male         63.9         7.0   \n",
              "2         3    ketan   0.416667    male         65.9         7.5   \n",
              "3         4    akash   0.500000    male         67.6         7.9   \n",
              "4         5     syam   0.583333    male         69.2         8.3   \n",
              "...     ...      ...        ...     ...          ...         ...   \n",
              "1671   1672    kiran  81.000000  female        155.0         NaN   \n",
              "1672   1673     mira  82.000000  female        155.0        65.0   \n",
              "1673   1674    radha  83.000000  female        155.0        65.0   \n",
              "1674   1675  lakshmi  84.000000  female        155.0        65.0   \n",
              "1675   1676   damini  85.000000  female        155.0        65.0   \n",
              "\n",
              "      Health Insurance cover  Premium  \n",
              "0                     500000     7015  \n",
              "1                     500000     7015  \n",
              "2                     500000     7015  \n",
              "3                     500000     7015  \n",
              "4                     500000     7015  \n",
              "...                      ...      ...  \n",
              "1671                10000000   111340  \n",
              "1672                10000000   111340  \n",
              "1673                10000000   111340  \n",
              "1674                10000000   111340  \n",
              "1675                10000000   111340  \n",
              "\n",
              "[1676 rows x 8 columns]"
            ]
          },
          "execution_count": 74,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df = pd.read_excel('height_weight_data.xlsx')\n",
        "df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LEXDnU0C5e4w"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CSG1wBCz5Vy7"
      },
      "source": [
        "3. Exploratory Data Analysis:\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AiK4yYLb5qij"
      },
      "source": [
        "1. Checking Missing values:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "difKXsU-4m--",
        "outputId": "f6b3d7e5-bc33-44a1-8a7c-5fa0dec0d5b3"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Sr.no</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age(yrs)</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Gender</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Height (cm)</th>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Weight(kg)</th>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Premium</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ],
            "text/plain": [
              "Sr.no                     0\n",
              "Name                      0\n",
              "Age(yrs)                  0\n",
              "Gender                    0\n",
              "Height (cm)               7\n",
              "Weight(kg)                6\n",
              "Health Insurance cover    0\n",
              "Premium                   0\n",
              "dtype: int64"
            ]
          },
          "execution_count": 75,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6rAPo9C34tel"
      },
      "outputs": [],
      "source": [
        "#The dataset contains missing values in two independent variables.\n",
        "#There are 7 missing values in Height and 6 in Weight, which will be handled using mean or median imputation"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Zg0GRPsV6g2k"
      },
      "source": [
        "2.  Checking for object column in dataset to encode:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zuE9m0xwuiQc",
        "outputId": "d4ce066b-c7d4-421e-b93c-a8039e095f57"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 8 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Sr.no                   1676 non-null   int64  \n",
            " 1   Name                    1676 non-null   object \n",
            " 2   Age(yrs)                1676 non-null   float64\n",
            " 3   Gender                  1676 non-null   object \n",
            " 4   Height (cm)             1669 non-null   float64\n",
            " 5   Weight(kg)              1670 non-null   float64\n",
            " 6   Health Insurance cover  1676 non-null   int64  \n",
            " 7   Premium                 1676 non-null   int64  \n",
            "dtypes: float64(3), int64(3), object(2)\n",
            "memory usage: 104.9+ KB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kOxBww2A7EV9"
      },
      "outputs": [],
      "source": [
        "#The dataset contains two object-type columns: Gender and Name. Only the Gender column will be encoded."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sI4Ps_hX7KN5"
      },
      "source": [
        "3. detecting outliers in the datasets:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "2swZSewPuiXh",
        "outputId": "d194d2be-e692-42bf-8495-f1d51cf06485"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='Age(yrs)'>"
            ]
          },
          "execution_count": 77,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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+8fs2bNbt5dS4pprromdd945evbsGaNGjYpXX321ghN3Tps2bYqrr746hg4dGj179ozddtstfvjDH7b53QWdYq1KB3H//feX6urqcscdd5S//vWv5dxzzy39+vUrq1atqvRondZxxx1XpkyZUubPn1/mzp1bvvCFL5SGhoby1ltvte5z/vnnl/r6+jJ9+vQye/bscvjhh5cjjzyyglN3brNmzSq77rprOeCAA8q4ceNar7dOlbd69eoyZMiQcuaZZ5bnn3++LF68uDzxxBPl73//e+s+kyZNKrW1teXRRx8t8+bNK1/60pfK0KFDy/r16ys4eeczYcKE0r9///L444+XJUuWlAcffLD07t27/OIXv2jdpzOsVYcJhEMPPbQ0Nja2Xt60aVMZNGhQmThxYgWn4oPeeOONEhFlxowZpZRS1qxZU7p3714efPDB1n1eeeWVEhHlueeeq9SYnVZzc3PZY489ypNPPlmOOeaY1kCwTh3D5ZdfXj7zmc986PaWlpYycODAcv3117det2bNmlJTU1Puu+++rTEi/8/o0aPLWWed1ea6r3zlK+W0004rpXSeteoQ32LYuHFjzJkzJ0aNGtV6XdeuXWPUqFHx3HPPVXAyPmjt2rUREfGJT3wiIiLmzJkT7777bpt1GzZsWDQ0NFi3CmhsbIzRo0e3WY8I69RRTJs2LUaMGBFf/epXY6eddorhw4fH7bff3rp9yZIlsXLlyjbrVFtbG4cddph12sqOPPLImD59eixatCgiIubNmxczZ86ME044ISI6z1pt9i9r2pL+9a9/xaZNm6Kurq7N9XV1dfG3v/2tQlPxQS0tLXHxxRfHyJEjY7/99ouIiJUrV0Z1dXX069evzb51dXWxcuXKCkzZed1///3xl7/8JV544YW0zTp1DIsXL45bbrklLr300rjqqqvihRdeiIsuuiiqq6tjzJgxrWvxvz0PWqet64orroimpqYYNmxYdOvWLTZt2hQTJkyI0047LSKi06xVhwgEOr7GxsaYP39+zJw5s9Kj8P9Zvnx5jBs3Lp588sno0aNHpcfhQ7S0tMSIESPixz/+cUREDB8+PObPnx+33nprjBkzpsLT8UEPPPBA3HPPPXHvvffGvvvuG3Pnzo2LL744Bg0a1KnWqkN8i2HAgAHRrVu3dFb1qlWrYuDAgRWaivddeOGF8fjjj8fTTz/d5ld8Dxw4MDZu3Bhr1qxps79127rmzJkTb7zxRhx00EFRVVUVVVVVMWPGjLjhhhuiqqoq6urqrFMHsPPOO8c+++zT5rq99947li1bFhHRuhaeByvve9/7XlxxxRVx6qmnxv777x9nnHFGXHLJJTFx4sSI6Dxr1SECobq6Og4++OCYPn1663UtLS0xffr0OOKIIyo4WedWSokLL7wwHnnkkXjqqadi6NChbbYffPDB0b179zbrtnDhwli2bJl124qOPfbYePnll2Pu3LmtXyNGjIjTTjut9c/WqfJGjhyZfkx40aJFMWTIkIiIGDp0aAwcOLDNOjU1NcXzzz9vnbaydevWRdeubV8eu3XrFi0tLRHRidaq0mdJvu/+++8vNTU15c477ywLFiwo5513XunXr19ZuXJlpUfrtC644IJSW1tbnnnmmfL666+3fq1bt651n/PPP780NDSUp556qsyePbscccQR5Ygjjqjg1JRS2vwUQynWqSOYNWtWqaqqKhMmTCivvvpqueeee8oOO+xQ7r777tZ9Jk2aVPr161cee+yx8tJLL5Uvf/nL292Pzm0LxowZU3bZZZfWH3N8+OGHy4ABA8pll13Wuk9nWKsOEwillHLjjTeWhoaGUl1dXQ499NDy5z//udIjdWoR8b9+TZkypXWf9evXl29/+9tlxx13LDvssEM56aSTyuuvv165oSml5ECwTh3Db37zm7LffvuVmpqaMmzYsHLbbbe12d7S0lKuvvrqUldXV2pqasqxxx5bFi5cWKFpO6+mpqYybty40tDQUHr06FE+9alPlfHjx5cNGza07tMZ1sqvewYAkg5xDgIA0LEIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCkCxcuDAGDhwYzc3N7XaMBQsWxODBg+Ptt99ut2MAm08gwDbmueeei27dusXo0aPb7RhXXnllfOc734k+ffq02zH22WefOPzww+NnP/tZux0D2Hw+ahm2Meecc0707t07Jk+eHAsXLoxBgwZt0ftftmxZ7L777rFkyZLYZZddNvt+3n333ejevfu/3ee3v/1tnHvuubFs2bKoqqra7GMBW553EGAb8tZbb8XUqVPjggsuiNGjR8edd97ZZvu0adNijz32iB49esRnP/vZuOuuu6JLly6xZs2a1n1mzpwZRx11VPTs2TPq6+vjoosuavM2/wMPPBAHHnhgaxy8/fbb0bdv33jooYfaHOvRRx+NXr16RXNzc7z22mvRpUuXmDp1ahxzzDHRo0ePuOeee2Lp0qXxxS9+MXbcccfo1atX7LvvvvG73/2u9T4+97nPxerVq2PGjBlb/sECPhaBANuQBx54IIYNGxZ77bVXnH766XHHHXfE+28CLlmyJE455ZQ48cQTY968eTF27NgYP358m9v/4x//iOOPPz5OPvnkeOmll2Lq1Kkxc+bMuPDCC1v3efbZZ2PEiBGtl3v16hWnnnpqTJkypc19TZkyJU455ZQ234a44oorYty4cfHKK6/EcccdF42NjbFhw4b44x//GC+//HL85Cc/id69e7fuX11dHZ/+9Kfj2Wef3aKPE7AFVPR3SQL/lSOPPLL8/Oc/L6WU8u6775YBAwaUp59+upRSyuWXX17222+/NvuPHz++RER58803SymlnH322eW8885rs8+zzz5bunbt2vp77A888MDygx/8oM0+zz//fOnWrVtZsWJFKaWUVatWlaqqqvLMM8+UUkpZsmRJiYjW2d63//77l+uuu+7f/p1OOumkcuaZZ37ERwDYWryDANuIhQsXxqxZs+Ib3/hGRERUVVXF17/+9Zg8eXLr9kMOOaTNbQ499NA2l+fNmxd33nln9O7du/XruOOOi5aWlliyZElERKxfvz569OiR7mffffeNu+66KyIi7r777hgyZEgcffTRbfb74DsPEREXXXRR/OhHP4qRI0fGtddeGy+99FL6e/Xs2TPWrVv33z4cQDsTCLCNmDx5crz33nsxaNCgqKqqiqqqqrjlllvi17/+daxdu/Yj3cdbb70VY8eOjblz57Z+zZs3L1599dXYbbfdIiJiwIAB8eabb6bbnnPOOa3nPEyZMiW+9a1vRZcuXdrs06tXr3SbxYsXxxlnnBEvv/xyjBgxIm688cY2+6xevTo++clPftSHAdhKBAJsA95777345S9/GT/96U/Ti/ugQYPivvvui7322itmz57d5nYvvPBCm8sHHXRQLFiwIHbffff0VV1dHRERw4cPjwULFqQZTj/99Fi6dGnccMMNsWDBghgzZsxHmr2+vj7OP//8ePjhh+O73/1u3H777W22z58/P4YPH/7fPBzA1lDp73EA/9kjjzxSqqury5o1a9K2yy67rIwYMaIsXry4dO/evVx22WVl4cKFZerUqWXw4MElIlpvN2/evNKzZ8/S2NhYXnzxxbJo0aLy6KOPlsbGxtb7mzZtWtlpp53Ke++9l471zW9+s1RXV5fjjz++zfXvn4Pw4osvtrl+3Lhx5fe//31ZvHhxmTNnTjnssMPK1772tTa369KlS3nttdc+zsMDtAPvIMA2YPLkyTFq1Kiora1N204++eSYPXt2NDc3x0MPPRQPP/xwHHDAAXHLLbe0/hRDTU1NREQccMABMWPGjFi0aFEcddRRMXz48LjmmmvafJbCCSecEFVVVfGHP/whHevss8+OjRs3xllnnfWR5t60aVM0NjbG3nvvHccff3zsueeecfPNN7duv+++++Lzn/98DBky5L96PID254OSYDs2YcKEuPXWW2P58uX/1e1uuummmDZtWjzxxBNtrv/Vr34Vl1xySaxYsaL1WxKba+PGjbHHHnvEvffeGyNHjvxY9wVseT66DLYjN998cxxyyCHRv3//+NOf/hTXX399m884+KjGjh0ba9asiebm5ujTp0+sW7cuXn/99Zg0aVKMHTv2Y8dBxP/9xMarrrpKHEAH5R0E2I5ccsklMXXq1Fi9enU0NDTEGWecEVdeeeXH/hjj6667LiZMmBBHH310PPbYY20+7AjYPgkEACBxkiIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBI/g9JDQs03LXQiQAAAABJRU5ErkJggg==",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Using Boxplot:\n",
        "\n",
        "sns.boxplot(x=df['Age(yrs)'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EJWfbhdUuiaB"
      },
      "outputs": [],
      "source": [
        "#using Zscore:\n",
        "#z_score = (Xi - Xmean)/Xstd"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SWAC9vq0uic3",
        "outputId": "f1db862f-901e-445f-e098-ac670de3e453"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "<>:5: SyntaxWarning: invalid decimal literal\n",
            "<>:5: SyntaxWarning: invalid decimal literal\n",
            "/tmp/ipykernel_17163/529088025.py:5: SyntaxWarning: invalid decimal literal\n",
            "  if zscore>3or zscore<-3:\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "[]"
            ]
          },
          "execution_count": 78,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "def zscore_age(col):\n",
        "  outliers =[]\n",
        "  for i in col:\n",
        "    zscore =(i-col.mean())/col.std()\n",
        "    if zscore>3or zscore<-3:\n",
        "      outliers.append(i)\n",
        "  return outliers\n",
        "zscore_age(df['Age(yrs)'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "12c7itYVuifR"
      },
      "outputs": [],
      "source": [
        "# we are not having outliers in age column."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VWCzvaXduihw",
        "outputId": "3a854a3f-35b9-4f89-f05a-3e0fe30a9c40"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Lower tale : 137.00000000000003\n",
            "Upper tale : 181.79999999999998\n"
          ]
        }
      ],
      "source": [
        "#Using IQR:\n",
        "\n",
        "q1 = df['Height (cm)'].quantile(.25)\n",
        "q2 = df['Height (cm)'].quantile(.50)\n",
        "q3 = df['Height (cm)'].quantile(.75)\n",
        "\n",
        "iqr = q3-q1\n",
        "lower_tale = q1-1.5*iqr\n",
        "upper_tale = q3+1.5*iqr\n",
        "\n",
        "print(\"Lower tale :\",lower_tale)\n",
        "print(\"Upper tale :\",upper_tale)\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "qEIvG60H-qeT",
        "outputId": "2ce183f4-84b0-44aa-8402-2c7c9f139430"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='Height (cm)'>"
            ]
          },
          "execution_count": 80,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgoAAAGwCAYAAADIeE3bAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAKxVJREFUeJzt3XtcFXX+x/H3OdyRuGmCqICaSlKWbqWgaT1WLX9ruWWu+bB+rZe00jVtba2fSb92NatNK8sbZduW9HM1L6mVZXk30i11XdP1UnhFMENAFBQ98/uDx5kY4KuoKKCv5+PhQzgz853vfM73zLzPzJyDy7IsSwAAABVwV3cHAABAzUVQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYOR7oQt6PB5lZmbqmmuukcvlqso+AQCAS8SyLB07dkwxMTFyu899vuCCg0JmZqYaN258oYsDAIBqtH//fjVq1Oic811wULjmmmvsFYWGhl5oMwAA4DLKz89X48aN7eP4uVxwUPBebggNDSUoAABQy1T2tgFuZgQAAEYEBQAAYERQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYERQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYERQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYERQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYERQAAAARgQFAABgRFAAAABGBAUAAGBEUAAAAEYEBQAAYERQAAAARr7V3QEAQO2WnZ2tvLy86u7GOYWFhSkqKqq6u1HrEBQAABcsOztbDz383yo+dbK6u3JOfv4BmvXB+4SF80RQAABcsLy8PBWfOqnCpp3lCQyTuzBXQRmrVdikkzxB4dXdPZu7KE/6cZXy8vIICueJoAAAuGiewDB56tT75fegcMfvqL24mREAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAKpIUVGRdu7cqaKiouruCmq5mjSWCAoAUEX27dunwYMHa9++fdXdFdRyNWksERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABj5VncHSvvpp580bNgw5eXlKSwsTG+99ZauvfZa4/Tx48drzJgxysvLU2hoqGJjY/Xzzz8rJiZGzz77rEJCQrR79249+uijsixLktSjRw9FRETo5ptv1s033ywfHx9lZWXp8ccfV35+vgICAjRgwAC99957KiwslL+/v4KDg3Xs2DEFBgZq5MiRCg0N1ZEjR7R582atWLFCRUVFkqSgoCCNHDlSO3bs0LJly1RcXKxrr71WUVFROnTokA4dOqQzZ85IkurWrav27dvrk08+sbeve/fuOnDggHbt2iXLshQWFqZ27drpq6++UlFRkTwej1wulwICAtS9e3eFhISodevWcrvdOnLkiLZv3y5JCgkJ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3ZUaj1pVDnWqPGpVOdSp8qhV5VzqOl3wzYwAAODKd0WdUQAAAFWLoAAAAIwICgAAwIigAAAAjGptUDh48KAeeugh1a1bV0FBQbrxxhv17bff2tMty1JKSooaNGigoKAgdenSRbt27arGHl9+Z86c0dixY9WkSRMFBQWpWbNm+stf/uL4fu+rtU6rV6/WPffco5iYGLlcLi1cuNAxvTJ1ycnJUb9+/RQaGqrw8HANHDhQBQUFl3ErLr2z1am4uFijR4/WjTfeqDp16igmJkb//d//rczMTEcbV0OdpHOPqdIee+wxuVwuvf76647Hr4ZaVaZO27dv17333quwsDDVqVNHt956q/bt22dPLyoq0tChQ1W3bl2FhISoV69eys7OvoxbcXmcq1YFBQUaNmyYGjVqpKCgILVq1UrTp093zFMVtaqVQeHo0aPq0KGD/Pz89Nlnn2nbtm2aOHGiIiIi7HleeeUVTZ48WdOnT9f69etVp04d3XXXXSoqKqrGnl9eL7/8sqZNm6a33npL27dv18svv6xXXnlFb775pj3P1Vqn48eP66abbtKUKVMqnF6ZuvTr10/ff/+9li1bpiVLlmj16tUaPHjw5dqEy+JsdTpx4oQ2btyosWPHauPGjZo/f7527Nihe++91zHf1VAn6dxjymvBggX65ptvFBMTU27a1VCrc9Xphx9+UMeOHZWQkKCVK1dqy5YtGjt2rAIDA+15Ro4cqcWLF2vu3LlatWqVMjMzdf/991+uTbhszlWrp556SkuXLtWsWbO0fft2jRgxQsOGDdOiRYvseaqkVlYtNHr0aKtjx47G6R6Px4qOjrb++te/2o/l5uZaAQEB1v/93/9dji7WCL/5zW+sAQMGOB67//77rX79+lmWRZ28JFkLFiywf69MXbZt22ZJsv75z3/a83z22WeWy+WyDh48eNn6fjmVrVNFNmzYYEmy9u7da1nW1VknyzLX6sCBA1bDhg2trVu3WnFxcdZrr71mT7saa1VRnfr06WM99NBDxmVyc3MtPz8/a+7cufZj27dvtyRZ6enpl6qr1a6iWiUmJlp//vOfHY+1bdvWGjNmjGVZVVerWnlGYdGiRbrlllvUu3dv1a9fX23atNHbb79tT8/IyFBWVpa6dOliPxYWFqZ27dopPT29OrpcLZKTk/XVV19p586dkqR//etfWrt2rbp37y6JOplUpi7p6ekKDw/XLbfcYs/TpUsXud1urV+//rL3uabIy8uTy+VSeHi4JOpUmsfj0cMPP6ynn35aiYmJ5aZTq5IaffLJJ2rRooXuuusu1a9fX+3atXOccv/uu+9UXFzseH0mJCQoNjb2qttvJScna9GiRTp48KAsy9KKFSu0c+dOdevWTVLV1apWBoUff/xR06ZNU/PmzfX555/r8ccf1/Dhw/X3v/9dkpSVlSVJioqKciwXFRVlT7saPPPMM3rwwQeVkJAgPz8/tWnTRiNGjFC/fv0kUSeTytQlKytL9evXd0z39fVVZGTkVVu7oqIijR49Wn379rX/MA11+sXLL78sX19fDR8+vMLp1Eo6fPiwCgoK9NJLL+nuu+/WF198ofvuu0/333+/Vq1aJamkTv7+/nYY9boa91tvvvmmWrVqpUaNGsnf31933323pkyZok6dOkmqulpd8F+PrE4ej0e33HKLXnzxRUlSmzZttHXrVk2fPl2PPPJINfeu5pgzZ47S0tL04YcfKjExUZs3b9aIESMUExNDnVCliouL9bvf/U6WZWnatGnV3Z0a57vvvtMbb7yhjRs3yuVyVXd3aiyPxyNJ6tmzp0aOHClJuvnmm/X1119r+vTp6ty5c3V2r8Z588039c0332jRokWKi4vT6tWrNXToUMXExDjOIlysWnlGoUGDBmrVqpXjseuvv96+KzY6OlqSyt3ZmZ2dbU+7Gjz99NP2WYUbb7xRDz/8sEaOHKkJEyZIok4mlalLdHS0Dh8+7Jh++vRp5eTkXHW184aEvXv3atmyZY4/c0udSqxZs0aHDx9WbGysfH195evrq7179+qPf/yj4uPjJVErSapXr558fX3PuX8/deqUcnNzHfNcbfutwsJC/c///I8mTZqke+65R61bt9awYcPUp08fvfrqq5Kqrla1Mih06NBBO3bscDy2c+dOxcXFSZKaNGmi6OhoffXVV/b0/Px8rV+/XklJSZe1r9XpxIkTcrudT7GPj4+d2qlTxSpTl6SkJOXm5uq7776z51m+fLk8Ho/atWt32ftcXbwhYdeuXfryyy9Vt25dx3TqVOLhhx/Wli1btHnzZvtfTEyMnn76aX3++eeSqJUk+fv769Zbbz3r/v1Xv/qV/Pz8HK/PHTt2aN++fVfVfqu4uFjFxcVn3cdXWa0u9A7M6rRhwwbL19fXGj9+vLVr1y4rLS3NCg4OtmbNmmXP89JLL1nh4eHWxx9/bG3ZssXq2bOn1aRJE6uwsLAae355PfLII1bDhg2tJUuWWBkZGdb8+fOtevXqWX/605/sea7WOh07dszatGmTtWnTJkuSNWnSJGvTpk323fqVqcvdd99ttWnTxlq/fr21du1aq3nz5lbfvn2ra5MuibPV6dSpU9a9995rNWrUyNq8ebN16NAh+9/JkyftNq6GOlnWucdUWWU/9WBZV0etzlWn+fPnW35+flZqaqq1a9cu680337R8fHysNWvW2G089thjVmxsrLV8+XLr22+/tZKSkqykpKTq2qRL5ly16ty5s5WYmGitWLHC+vHHH62//e1vVmBgoDV16lS7jaqoVa0MCpZlWYsXL7ZuuOEGKyAgwEpISLBSU1Md0z0ejzV27FgrKirKCggIsH79619bO3bsqKbeVo/8/HzrySeftGJjY63AwECradOm1pgxYxw78au1TitWrLAklfv3yCOPWJZVubr8/PPPVt++fa2QkBArNDTU6t+/v3Xs2LFq2JpL52x1ysjIqHCaJGvFihV2G1dDnSzr3GOqrIqCwtVQq8rUaebMmdZ1111nBQYGWjfddJO1cOFCRxuFhYXWE088YUVERFjBwcHWfffdZx06dOgyb8mld65aHTp0yPr9739vxcTEWIGBgVbLli2tiRMnWh6Px26jKmrFn5kGAABGtfIeBQAAcHkQFAAAgBFBAQAAGBEUAACAEUEBAAAYERQAAIARQQEAABgRFAAAgBFBAbgCxMfH6/XXX6/0/Hv27JHL5dLmzZurrA+dOnXShx9+WGXtVaR9+/aaN2/eJV0HACeCAlBNfv/73+u3v/1tucdXrlwpl8tV7i++nc0///lPDR48uOo6J+m9994r93fsTRYtWqTs7Gw9+OCDVdqHsp577jk988wz9h+9AXDpERSAK8C1116r4ODgalv/5MmT1b9//3J/ya6qde/eXceOHdNnn312SdcD4BcEBaAWWLt2rW6//XYFBQWpcePGGj58uI4fP25PL3vp4T//+Y86duyowMBAtWrVSl9++aVcLpcWLlzoaPfHH3/UnXfeqeDgYN10001KT0+XVHJWo3///srLy5PL5ZLL5dL//u//Vti3n376ScuXL9c999zjeDw3N1dDhgxRVFSUAgMDdcMNN2jJkiWSfjlbsWTJErVs2VLBwcF64IEHdOLECf39739XfHy8IiIiNHz4cJ05c8Zu08fHR//1X/+l2bNnX0Q1AZwPggJQw/3www+6++671atXL23ZskX/+Mc/tHbtWg0bNqzC+c+cOaPf/va3Cg4O1vr165WamqoxY8ZUOO+YMWM0atQobd68WS1atFDfvn11+vRpJScn6/XXX1doaKgOHTqkQ4cOadSoURW2sXbtWgUHB+v666+3H/N4POrevbvWrVunWbNmadu2bXrppZfk4+Njz3PixAlNnjxZs2fP1tKlS7Vy5Urdd999+vTTT/Xpp5/qgw8+0IwZM/TRRx851nfbbbdpzZo151tGABfIt7o7AFzNlixZopCQEMdjpd9BS9KECRPUr18/jRgxQpLUvHlzTZ48WZ07d9a0adMUGBjomH/ZsmX64YcftHLlSkVHR0uSxo8fr65du5Zb/6hRo/Sb3/xGkvTCCy8oMTFRu3fvVkJCgsLCwuRyuew2TPbu3auoqCjHZYcvv/xSGzZs0Pbt29WiRQtJUtOmTR3LFRcXa9q0aWrWrJkk6YEHHtAHH3yg7OxshYSEqFWrVrrzzju1YsUK9enTx14uJiZG+/fvl8fjueSXOgAQFIBqdeedd2ratGmOx9avX6+HHnrI/v1f//qXtmzZorS0NPsxy7Lk8XiUkZHheCcvSTt27FDjxo0dB/jbbrutwvW3bt3a/rlBgwaSpMOHDyshIaHS21BYWFgurGzevFmNGjWyQ0JFgoOD7ZAgSVFRUYqPj3cEp6ioKB0+fNixXFBQkDwej06ePKmgoKBK9xPAhSEoANWoTp06uu666xyPHThwwPF7QUGBhgwZouHDh5dbPjY29qLW7+fnZ//scrkk6bw/UVCvXj0dPXrU8VhlDuCl1+1df0WPle1PTk6O6tSpQ0gALhOCAlDDtW3bVtu2bSsXKExatmyp/fv3Kzs7W1FRUZJKPj55vvz9/ctdBqlImzZtlJWVpaNHjyoiIkJSyZmKAwcOaOfOnWc9q3Ahtm7dqjZt2lRpmwDMuMAH1HCjR4/W119/rWHDhmnz5s3atWuXPv74Y+PNjF27dlWzZs30yCOPaMuWLVq3bp2ee+45Sb+cNaiM+Ph4FRQU6KuvvtKRI0d04sSJCudr06aN6tWrp3Xr1tmPde7cWZ06dVKvXr20bNkyZWRk6LPPPtPSpUvPY8srtmbNGnXr1u2i2wFQOQQFoIZr3bq1Vq1apZ07d+r2229XmzZtlJKSopiYmArn9/Hx0cKFC1VQUKBbb71VgwYNsj/1UPZegrNJTk7WY489pj59+ujaa6/VK6+8Ylxf//79HfdQSNK8efN06623qm/fvmrVqpX+9Kc/VeoMxdkcPHhQX3/9tfr3739R7QCoPJdlWVZ1dwLApbVu3Tp17NhRu3fvdtxAWFWysrKUmJiojRs3Ki4ursrb9xo9erSOHj2q1NTUS7YOAE7cowBcgRYsWKCQkBA1b95cu3fv1pNPPqkOHTpckpAgSdHR0Zo5c6b27dt3SYNC/fr19dRTT12y9gGUxxkF4Ar0/vvva9y4cdq3b5/q1aunLl26aOLEiapbt251dw1ALUNQAAAARtzMCAAAjAgKAADAiKAAAACMCAoAAMCIoAAAAIwICgAAwIigAAAAjAgKAADA6P8BJyfCQTPSZJkAAAAASUVORK5CYII=",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.boxplot(x=df['Height (cm)'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LgTlExfZuikd"
      },
      "outputs": [],
      "source": [
        "# We are having outliers in Height column,we'll handle these outliers in feature engineering."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FkAW1saWuinK",
        "outputId": "10b6bc3c-aabb-45d1-a838-430aa66bb7de"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 8 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Sr.no                   1676 non-null   int64  \n",
            " 1   Name                    1676 non-null   object \n",
            " 2   Age(yrs)                1676 non-null   float64\n",
            " 3   Gender                  1676 non-null   object \n",
            " 4   Height (cm)             1669 non-null   float64\n",
            " 5   Weight(kg)              1670 non-null   float64\n",
            " 6   Health Insurance cover  1676 non-null   int64  \n",
            " 7   Premium                 1676 non-null   int64  \n",
            "dtypes: float64(3), int64(3), object(2)\n",
            "memory usage: 104.9+ KB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "id": "GaV09lATuiqJ",
        "outputId": "8a0f4c82-3d1f-4459-84aa-1826658b75e2"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: ylabel='Weight(kg)'>"
            ]
          },
          "execution_count": 82,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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nVasWKEhQ4bI4XCoR48euvjii9WtWzdlZGRo7Nix2rVrV1vOCgAAECSkkFm8eLF69OihwsJC5eXl6d1339Wnn36qffv2aceOHZozZ46OHz+uG264QTfeeKP279/f1nMDAACE9tbSrl27tH37dl1yySUt7r/mmmt03333aeXKlSosLNT777+vrKysiA4KAADwYyGFzGuvvRbSk8XGxmrChAn/00AAAACh4lNLAADAWGF/aulXv/qVbDbbadttNpvsdrt69eqlO++8U3369InIgAAAAK0J+4iM0+nU1q1bVV5eLpvNJpvNpk8++URbt27V8ePH9frrr+vyyy/XBx980BbzAgAABIR9RMbtduvOO+/U0qVLFRNzooN8Pp8efPBBJSQkaP369ZowYYIeffRRlZaWRnxgAACAk8I+IvPiiy9qypQpgYiRpJiYGE2ePFkFBQWy2WyaNGmSvvjii4gOCgAA8GNhh8zx48f11Vdfnbb9q6++UnNzsyTJbre3eB4NAABAJIX91tLdd9+t+++/X48//rgGDBgg6cR1ZubPn6/f/e53kqSSkpJWrzkDAAAQKWGHzJIlS+RyubRo0SLV1NRIklwul6ZOnapHH31UkgJX+AUAAGhLYYfMeeedp5kzZ2rmzJnyer2SJIfDEbQmPT09MtMBAACcQdjnyJx6lV+HwxEUMQ8//HBkpgIAAAhB2CEzceJEbdy48bTtU6dO1auvvhqRoQAAAEIRdsisXbtWv/3tb4OuETN58mS98cYbKi4ujuhwAAAAZxJ2yNx8881avny5fvnLX2r37t164IEH9Pbbb6u4uFjZ2dltMSMAAECLwj7ZV5LuvPNO1dbWKicnR926dVNJSYl69eoV6dkAAADOKKSQmTZtWovbu3Xrpv79+2v58uWBbYsXL47MZAAAAD8hpJD55JNPWtzeq1cveb3ewH6u5gsAAM6lkEKGk3gBAEA0CvtkXwAAgGgRUshMmDBB//73v0N6wtdff11r1679n4YCAAAIRUhvLXXr1k2XXHKJcnJydOutt+rqq69Wamqq7Ha7Dh8+rD179qi0tFTr169XamqqCgoK2npuAACA0ELm6aef1qRJk/SXv/xFy5cv1549e4L2JyQkKC8vTwUFBdwsEgAAnDMhX0fG5XIFbhZ5+PBhVVZW6ocfftDPfvYz9ezZk08sAQCAcy7sC+JVVlYqLS1NSUlJLe7jztcAAOBcCftTS5mZmfr+++9P2/7f//5XmZmZERkKAAAgFGGHjN/vb/FtpCNHjshut0dkKAAAgFCE/NbSydsU2Gw2zZo1SxdccEFgX3Nzs8rKynTFFVdEfEAAAIDWhBwyJ29D4Pf79fnnn6tz586BfZ07d9bll1+u6dOnR35CAACAVoQcMidvU3Dvvffqz3/+sxwOR5sNhdbFNHisHgEAEMU62s+JsD+1VFhY2BZz4Cc4nU516hwrfVNi9SgAgCjXqXOsnE6n1WOcE2GHTH19vZ555hlt2bJF3333nXw+X9D+b775JmLD4f9zuVx69ZWX5fF0rNIGzuTgwYOaN2+eZs6cqYyMDKvHAaKG0+mUy+WyeoxzIuyQ+f3vf6+SkhLdfffdSklJ4UJ455DL5eow/2EC4cjIyFDv3r2tHgOABcIOmY0bN+qf//yncnJy2mIeAACAkIV9HZmkpCQlJye3xSwAAABhCTtknn76ac2ePVtHjx5ti3kAAABCFtJbS1deeWXQuTBff/21XC6XevTooU6dOgWtLS8vj+yEAAAArQgpZEaOHNnGYwAAAIQvpJCZM2dOW88BAAAQtrDPkQEAAIgWYX/8OikpqcVrx9hsNtntdvXq1UtjxozRvffeG5EBAQAAWhN2yMyePVvz5s3TTTfdpGuuuUaS9NFHH6moqEj5+fk6cOCAJk6cqOPHj2vs2LERHxgAAOCksEOmtLRUc+fO1YQJE4K2r1q1Su+9957eeustXXbZZXr++ecJGQAA0KbCPkdm06ZNysvLO2378OHDtWnTJknSL37xC+65BAAA2lzYIZOcnKwNGzactn3Dhg2BK/7W19crISHhf58OAADgDMJ+a2nWrFmaOHGiiouLA+fI7Nq1S//617+0cuVKSdLmzZs1ZMiQyE4KAADwI2GHzNixY9W3b18tXbpUb7/9tiSpT58+Kikp0aBBgyRJDz30UGSnBAAAaEHYISNJOTk53P0aAABYLqSQ8Xq9cjgcga/P5OQ6AACAthZSyCQlJek///mPunfvrsTExBYviOf3+2Wz2dTc3BzxIQEAAFoSUshs3bo18Imk4uLiNh0IAAAgVCGFzKmfQOLTSAAAIFqc1U0j33//fd11110aNGiQvv32W0nSK6+8otLS0ogOBwAAcCZhh8xbb72lESNGKC4uTuXl5WpsbJQkeTwezZ8/P+IDAgAAtCbskJk7d65Wrlyp1atXq1OnToHtOTk5Ki8vP+tBnnnmGdlsNk2ZMiWwraGhQfn5+eratau6dOmiUaNGqaam5qxfAwAAtC9hh0xFRYUGDx582nan06na2tqzGmLXrl1atWqVLrvssqDtU6dO1YYNG/Tmm2+qpKREVVVVuv3228/qNQAAQPsTdsi43W59/fXXp20vLS3VRRddFPYAR44c0ejRo7V69WolJSUFtns8Hr344otavHixhg0bpquuukqFhYX68MMPtXPnzrBfBwAAtD9hh8zYsWP14IMPqqysTDabTVVVVVq7dq2mT5+uiRMnhj1Afn6+br755tPuqL179241NTUFbc/OzlZ6erp27NjR6vM1NjbK6/UGPQAAQPsU8i0KDhw4oMzMTD322GPy+XwaPny4jh49qsGDBys2NlbTp0/X5MmTw3rx9evXq7y8XLt27TptX3V1tTp37qzExMSg7S6XS9XV1a0+54IFC/Tkk0+GNQcAADBTyCHTs2dPZWRkKDc3V7m5udq7d6/q6up05MgR9e3bV126dAnrhQ8dOqQHH3xQmzdvlt1uD3vw1syYMUPTpk0LfO/1epWWlhax5wcAANEj5JDZunWrtm3bpm3btum1117TsWPHdNFFF2nYsGEaNmyYhg4dKpfLFfIL7969W99995369+8f2Nbc3Kzt27dr6dKl2rRpk44dO6ba2tqgozI1NTVyu92tPm9sbKxiY2NDngMAAJgr5JAZOnSohg4dKunEx6I//PDDQNisWbNGTU1Nys7O1pdffhnS8w0fPlyff/550LZ7771X2dnZevTRR5WWlqZOnTppy5YtGjVqlKQTn5iqrKzUwIEDQx0bAAC0YyGHzKnsdruGDRum66+/Xrm5udq4caNWrVqlr776KuTnSEhIUL9+/YK2xcfHq2vXroHt999/v6ZNm6bk5GQ5HA5NnjxZAwcO1HXXXXc2YwMAgHYmrJA5duyYdu7cqeLiYm3btk1lZWVKS0vT4MGDtXTp0ojfh2nJkiWKiYnRqFGj1NjYqBEjRmj58uURfQ0AAGCukENm2LBhKisrU2ZmpoYMGaLx48dr3bp1SklJidgw27ZtC/rebrdr2bJlWrZsWcReAwAAtB8hh8z777+vlJSUwIm9Q4YMUdeuXdtyNgAAgDMK+YJ4tbW1Kigo0AUXXKCFCxcqNTVVl156qSZNmqS//vWv+v7779tyTgAAgNOEfEQmPj5eN954o2688UZJUl1dnUpLS1VcXKxFixZp9OjRysrK0hdffNFmwwIAAJwq7FsUnBQfH6/k5GQlJycrKSlJ559/vvbu3RvJ2QAAAM4o5CMyPp9PH3/8sbZt26bi4mJ98MEHqq+v189//nPl5uZq2bJlys3NbctZAQAAgoQcMomJiaqvr5fb7VZubq6WLFmioUOHqmfPnm05HwAAQKtCDpk//vGPys3NVe/evdtyHgAAgJCFHDLjx49vyzkAAADCdtYn+wIAAFiNkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMayNGQWLFigAQMGKCEhQd27d9fIkSNVUVERtKahoUH5+fnq2rWrunTpolGjRqmmpsaiiQEAQDSxNGRKSkqUn5+vnTt3avPmzWpqatINN9yg+vr6wJqpU6dqw4YNevPNN1VSUqKqqirdfvvtFk4NAACixflWvnhRUVHQ9y+99JK6d++u3bt3a/DgwfJ4PHrxxRe1bt06DRs2TJJUWFioiy++WDt37tR1111nxdgAACBKRNU5Mh6PR5KUnJwsSdq9e7eampqUl5cXWJOdna309HTt2LGjxedobGyU1+sNegAAgPYpakLG5/NpypQpysnJUb9+/SRJ1dXV6ty5sxITE4PWulwuVVdXt/g8CxYskNPpDDzS0tLaenQAAGCRqAmZ/Px8ffHFF1q/fv3/9DwzZsyQx+MJPA4dOhShCQEAQLSx9ByZkyZNmqR//OMf2r59uy688MLAdrfbrWPHjqm2tjboqExNTY3cbneLzxUbG6vY2Ni2HhkAAEQBS4/I+P1+TZo0Se+88462bt2qzMzMoP1XXXWVOnXqpC1btgS2VVRUqLKyUgMHDjzX4wIAgChj6RGZ/Px8rVu3Tn/729+UkJAQOO/F6XQqLi5OTqdT999/v6ZNm6bk5GQ5HA5NnjxZAwcO5BNLAADA2pBZsWKFJGno0KFB2wsLCzVmzBhJ0pIlSxQTE6NRo0apsbFRI0aM0PLly8/xpAAAIBpZGjJ+v/8n19jtdi1btkzLli07BxMBAACTRM2nlgAAAMJFyAAAAGMRMgAAwFiEDAAAMBYhAwAAjEXIAAAAYxEyAIzU3NysiooKSSeu+N3c3GzxRACsYPOHcjEXg3m9XjmdTnk8HjkcDqvHARAB27dv1/LlywNXA5dO3JvtgQce0ODBgy2cDECkhPrzOypuGgmYpKGhQZWVlVaP0WGVl5dr1apV6tevn9LS0rRr1y4NGDBADQ0NmjNnjsaPH6/+/ftbPWaHlZ6eLrvdbvUY6EA4IgOEad++fRo3bpzVYwBRqaCgQL1797Z6DLQDHJEB2kh6eroKCgqsHqNDqqio0LPPPvuT6x566CH16dPnHEyEH0tPT7d6BHQwhAwQJrvdzp84LXLgwIGQ1nXu3Jl/R0AHwaeWABjjyy+/jOg6AOYjZAAYo6qqKqLrAJiPt5YAGOOzzz4LfB0fH6+srCz5fD7FxMRo//79qq+vP20dgPaNkAFgjKampsDX9fX1+vTTT39yHYD2jbeWABirT58+uueee/iEEtCBcUQGgDEuuOACHT16NPB9RUVF4DYFP14HoGPgiAwAY/Tt2zei6wCYj5ABYIxQL0Tezi9YDuAUhAwAY3Tr1i2i6wCYj5ABYIzy8vKIrgNgPkIGgDFqa2sjug6A+QgZAMaw2+0RXQfAfIQMAGNkZWUFvu7Tp49iYk78LywmJiboWjKnrgPQvnEdGQDG8Hq9ga9PvX6Mz+cL+v7UdQDaN47IADCGzWaL6DoA5iNkABgjOzs7ousAmI+QAWCM9957L+j7xMREZWVlKTEx8YzrALRfnCMDwBg/vqt1bW1tix+15u7XQMfBERkAAGAsQgaAMU79iHUk1gEwHyEDAACMRcgAMIbH44noOgDmI2QAGOPbb7+N6DoA5iNkAACAsQgZAABgLEIGAAAYi5ABAADGImQAAICxCBkAAGAsQgYAABiLkAEAAMYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkABgjJia0/2WFug6A+fjdDsAYPp8vousAmI+QAQAAxiJkAACAsQgZAMZITEyM6DoA5iNkAACAsQgZAMaora2N6DoA5iNkAACAsQgZAMZISEgIfP3SSy/J5XLJbrfL5XLppZdeanEdgPaNkAFgjNWrVwe+HjNmjOx2ux5//HHZ7XaNGTOmxXUA2rfzrR4AAELldrtlt9vV0NAgSTp48KBmz54dtMZut8vtdlsxHgALcEQGgFGKiopkt9tb3Ge321VUVHSOJwJgJY7IADBOUVGRqqurNXHiRB05ckRdunTRihUrOBIDdECEDAAjud1uvfPOO1aPAcBivLUEAACMRcgAAABjETIAAMBYhAwAADAWIQMAAIxFyAAAAGMRMgAAwFiEDAAAMBYhAwAAjNXur+zr9/slSV6v1+JJAABAqE7+3D75c7w17T5k6urqJElpaWkWTwIAAMJVV1cnp9PZ6n6b/6dSx3A+n09VVVVKSEiQzWazehwAEeT1epWWlqZDhw7J4XBYPQ6ACPL7/aqrq1NqaqpiYlo/E6bdhwyA9svr9crpdMrj8RAyQAfFyb4AAMBYhAwAADAWIQPAWLGxsZozZ45iY2OtHgWARThHBgAAGIsjMgAAwFiEDAAAMBYhAwAAjEXIAAAAYxEyAADAWIQMAAAwFiEDAACMRcgAAABj/R/gfEmgwKwVKAAAAABJRU5ErkJggg==",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.boxplot(df['Weight(kg)'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ElL2ctqPuisz"
      },
      "outputs": [],
      "source": [
        "# We are having outliers in Weight column, we'll handle these outliers in feature engineering.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iEJgfnHXDLgL"
      },
      "source": [
        "4. Feature Engineering:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Lpv_67BUuiv-",
        "outputId": "510433c9-46a7-4ef3-b564-f7d17e664ef3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 8 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Sr.no                   1676 non-null   int64  \n",
            " 1   Name                    1676 non-null   object \n",
            " 2   Age(yrs)                1676 non-null   float64\n",
            " 3   Gender                  1676 non-null   object \n",
            " 4   Height (cm)             1669 non-null   float64\n",
            " 5   Weight(kg)              1670 non-null   float64\n",
            " 6   Health Insurance cover  1676 non-null   int64  \n",
            " 7   Premium                 1676 non-null   int64  \n",
            "dtypes: float64(3), int64(3), object(2)\n",
            "memory usage: 104.9+ KB\n"
          ]
        }
      ],
      "source": [
        "# Filling missing/null values :\n",
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aKedhqOvuiz4",
        "outputId": "ca11f232-1615-444f-8b9f-f7b2c47ab4c2"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/tmp/ipykernel_17163/757982864.py:5: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  df['Height (cm)'].fillna(df['Height (cm)'].median(), inplace=True)\n"
          ]
        }
      ],
      "source": [
        "df['Height (cm)'].isna().sum()\n",
        "\n",
        "df['Height (cm)'].median()\n",
        "\n",
        "df['Height (cm)'].fillna(df['Height (cm)'].median(), inplace=True)\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lK7g-8lDui3U",
        "outputId": "2064a765-1120-47ea-e149-78ba486f1652"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "np.int64(0)"
            ]
          },
          "execution_count": 85,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df['Height (cm)'].isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6DF-tWqNujYO",
        "outputId": "998d74fe-a4d4-441d-c3c8-8c6f498e41b3"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "np.int64(6)"
            ]
          },
          "execution_count": 32,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df['Weight(kg)'].isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        },
        "id": "EH91amA1uja5",
        "outputId": "9da612e9-53db-4e07-f833-c9b429ddb8c1"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/tmp/ipykernel_17163/3556953422.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  df['Weight(kg)'].fillna(df['Weight(kg)'].median(), inplace=True)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Sr.no</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age(yrs)</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Gender</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Height (cm)</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Weight(kg)</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Premium</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ],
            "text/plain": [
              "Sr.no                     0\n",
              "Name                      0\n",
              "Age(yrs)                  0\n",
              "Gender                    0\n",
              "Height (cm)               0\n",
              "Weight(kg)                0\n",
              "Health Insurance cover    0\n",
              "Premium                   0\n",
              "dtype: int64"
            ]
          },
          "execution_count": 86,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df['Weight(kg)'].fillna(df['Weight(kg)'].median(), inplace=True)\n",
        "\n",
        "df.isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hskHHtj_ujd6"
      },
      "outputs": [],
      "source": [
        "# Performing encoding over df['Gender'] feature :"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wx7Fod4Xujg8",
        "outputId": "69bb26c1-6dd3-47ee-abab-0d56e1764e34"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array(['male', 'female'], dtype=object)"
            ]
          },
          "execution_count": 87,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df['Gender'].unique()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "g092IBDLujjo",
        "outputId": "d842f984-2a26-42a7-e3ca-fe0553e211b5"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/tmp/ipykernel_17163/2855755960.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  df['Gender'].replace({'male':0, 'female':1}, inplace=True)\n",
            "/tmp/ipykernel_17163/2855755960.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
            "  df['Gender'].replace({'male':0, 'female':1}, inplace=True)\n"
          ]
        }
      ],
      "source": [
        "df['Gender'].replace({'male':0, 'female':1}, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Uw6AUyXMujmr",
        "outputId": "10d9b0b5-10e7-4295-d0dd-74511b83f6b3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 8 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Sr.no                   1676 non-null   int64  \n",
            " 1   Name                    1676 non-null   object \n",
            " 2   Age(yrs)                1676 non-null   float64\n",
            " 3   Gender                  1676 non-null   object \n",
            " 4   Height (cm)             1676 non-null   float64\n",
            " 5   Weight(kg)              1676 non-null   float64\n",
            " 6   Health Insurance cover  1676 non-null   int64  \n",
            " 7   Premium                 1676 non-null   int64  \n",
            "dtypes: float64(3), int64(3), object(2)\n",
            "memory usage: 104.9+ KB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "PgRbjs6RujqZ",
        "outputId": "1d00034c-3db1-442e-862b-db7dd213da7e"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1676,\n  \"fields\": [\n    {\n      \"column\": \"Sr.no\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 483,\n        \"min\": 1,\n        \"max\": 1676,\n        \"num_unique_values\": 1676,\n        \"samples\": [\n          865,\n          1269,\n          417\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 353,\n        \"samples\": [\n          \"Angel\",\n          \"akshay\",\n          \"Nathan\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age(yrs)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 26.338582349667547,\n        \"min\": 0.25,\n        \"max\": 85.0,\n        \"num_unique_values\": 94,\n        \"samples\": [\n          34.0,\n          16.0,\n          49.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Height (cm)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 31.364206676554616,\n        \"min\": 56.0,\n        \"max\": 177.0,\n        \"num_unique_values\": 161,\n        \"samples\": [\n          113.5,\n          67.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Weight(kg)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 22.932971692550783,\n        \"min\": 4.8,\n        \"max\": 88.0,\n        \"num_unique_values\": 274,\n        \"samples\": [\n          70.0,\n          79.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Health Insurance cover\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3170480,\n        \"min\": 500000,\n        \"max\": 10000000,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          7500000,\n          750000\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Premium\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 25308,\n        \"min\": 7015,\n        \"max\": 111340,\n        \"num_unique_values\": 82,\n        \"samples\": [\n          26410,\n          7015\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"BMI\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4.857805256764994,\n        \"min\": 13.770213967940117,\n        \"max\": 35.37981269510926,\n        \"num_unique_values\": 352,\n        \"samples\": [\n          24.99281815570238,\n          27.281746031746035\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "df"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-d21ef87f-b1f8-41a6-80f0-08a526e27d70\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sr.no</th>\n",
              "      <th>Name</th>\n",
              "      <th>Age(yrs)</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Height (cm)</th>\n",
              "      <th>Weight(kg)</th>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <th>Premium</th>\n",
              "      <th>BMI</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>krushna</td>\n",
              "      <td>0.250000</td>\n",
              "      <td>male</td>\n",
              "      <td>61.4</td>\n",
              "      <td>6.4</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>16.976307</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>pavan</td>\n",
              "      <td>0.333333</td>\n",
              "      <td>male</td>\n",
              "      <td>63.9</td>\n",
              "      <td>7.0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.143375</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>ketan</td>\n",
              "      <td>0.416667</td>\n",
              "      <td>male</td>\n",
              "      <td>65.9</td>\n",
              "      <td>7.5</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.269924</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>akash</td>\n",
              "      <td>0.500000</td>\n",
              "      <td>male</td>\n",
              "      <td>67.6</td>\n",
              "      <td>7.9</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.287560</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>syam</td>\n",
              "      <td>0.583333</td>\n",
              "      <td>male</td>\n",
              "      <td>69.2</td>\n",
              "      <td>8.3</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.332687</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1671</th>\n",
              "      <td>1672</td>\n",
              "      <td>kiran</td>\n",
              "      <td>81.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1672</th>\n",
              "      <td>1673</td>\n",
              "      <td>mira</td>\n",
              "      <td>82.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1673</th>\n",
              "      <td>1674</td>\n",
              "      <td>radha</td>\n",
              "      <td>83.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1674</th>\n",
              "      <td>1675</td>\n",
              "      <td>lakshmi</td>\n",
              "      <td>84.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1675</th>\n",
              "      <td>1676</td>\n",
              "      <td>damini</td>\n",
              "      <td>85.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>155.0</td>\n",
              "      <td>65.0</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1676 rows × 9 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d21ef87f-b1f8-41a6-80f0-08a526e27d70')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d21ef87f-b1f8-41a6-80f0-08a526e27d70 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d21ef87f-b1f8-41a6-80f0-08a526e27d70');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "  <div id=\"id_05da58a4-db86-490c-9c2c-f8fcb9dd618a\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('df')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_05da58a4-db86-490c-9c2c-f8fcb9dd618a button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('df');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "      Sr.no     Name   Age(yrs)  Gender  Height (cm)  Weight(kg)  \\\n",
              "0         1  krushna   0.250000    male         61.4         6.4   \n",
              "1         2    pavan   0.333333    male         63.9         7.0   \n",
              "2         3    ketan   0.416667    male         65.9         7.5   \n",
              "3         4    akash   0.500000    male         67.6         7.9   \n",
              "4         5     syam   0.583333    male         69.2         8.3   \n",
              "...     ...      ...        ...     ...          ...         ...   \n",
              "1671   1672    kiran  81.000000  female        155.0        65.0   \n",
              "1672   1673     mira  82.000000  female        155.0        65.0   \n",
              "1673   1674    radha  83.000000  female        155.0        65.0   \n",
              "1674   1675  lakshmi  84.000000  female        155.0        65.0   \n",
              "1675   1676   damini  85.000000  female        155.0        65.0   \n",
              "\n",
              "      Health Insurance cover  Premium        BMI  \n",
              "0                     500000     7015  16.976307  \n",
              "1                     500000     7015  17.143375  \n",
              "2                     500000     7015  17.269924  \n",
              "3                     500000     7015  17.287560  \n",
              "4                     500000     7015  17.332687  \n",
              "...                      ...      ...        ...  \n",
              "1671                10000000   111340  27.055151  \n",
              "1672                10000000   111340  27.055151  \n",
              "1673                10000000   111340  27.055151  \n",
              "1674                10000000   111340  27.055151  \n",
              "1675                10000000   111340  27.055151  \n",
              "\n",
              "[1676 rows x 9 columns]"
            ]
          },
          "execution_count": 89,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df['BMI'] = df['Weight(kg)']/((df['Height (cm)']/100)**2)\n",
        "\n",
        "df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "oTcen_usujt4",
        "outputId": "66937fdf-5f37-4ce6-bee4-9b7dda101eb8"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='BMI'>"
            ]
          },
          "execution_count": 90,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAggAAAGwCAYAAADMjZ3mAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAEStJREFUeJzt3W1onfX5wPHrxLaxmgdNWpOWplk1qIhWoYgrQnXr0FQUN30zRI1UFLWKD+C07sXwxajs3WBjsInK5jpHt1WdoNtwJmVQEYVQN0aHVZpKHxSxSVqbWJZ7L/4kLF6N7T+suW97Ph8ITc65T3K1v/zO+facu02tKIoiAAD+S0PZAwAA1SMQAIBEIAAAiUAAABKBAAAkAgEASAQCAJDMm+0NJyYmYu/evdHc3By1Wu1/ORMAcJIURRGjo6OxdOnSaGiY+XmCWQfC3r17o6ura7Y3BwBKtGfPnli2bNmM1886EJqbm6e+QEtLy2w/DQAwh0ZGRqKrq2vqcXwmsw6EyZcVWlpaBAIAfMUc7/QAJykCAIlAAAASgQAAJAIBAEgEAgCQCAQAIBEIAEAiEACARCAAAIlAAAASgQAAJAIBAEgEAgCQCAQAIBEIAEAiEACARCAAAIlAAAASgQAAJAIBAEgEAgCQCAQAIBEIAEAiEACARCAAAIlAAAASgQAAJAIBAEgEAgCQCAQAIBEIAEAiEACARCAAAMm8sgeAAwcOxPDwcNljQLS2tkZHR0fZY0AlCARKdeDAgbj1ttvj6OfjZY8CMX9BYzz/q1+KBAiBQMmGh4fj6OfjceTcq2Li9NayxzmlNBw5GAs/2BZHVqyJiYVnlT1O5TWMDUe8PxDDw8MCAUIgUBETp7fGxJmLyh7jlDSx8Cx/tsD/m5MUAYBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJBULhDGxsbiX//6V4yNjZU9CgCUogqPhZULhKGhobj77rtjaGio7FEAoBRVeCysXCAAAOUTCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkMw70QPHx8djfHx86uORkZGTMtCk3bt3n9TPTzVYZ6rG9yRVUIXvwxMOhE2bNsWTTz55MmeZ5oc//OGcfS2ASe574P+ccCBs3LgxHnnkkamPR0ZGoqur66QMFRHx/e9/P7q7u0/a56cadu/e7Q6ZSnHfQxVU4b7xhAOhsbExGhsbT+Ys03R3d8f5558/Z18PIMJ9D0xykiIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIKhcIy5cvj5///OexfPnyskcBgFJU4bFwXmlfeQann356nH/++WWPAQClqcJjYeWeQQAAyicQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASOaVPQBERDSMDZc9wimn4cjBab/y5XwPwnQCgVK1trbG/AWNEe8PlD3KKWvhB9vKHuErY/6CxmhtbS17DKgEgUCpOjo64vlf/TKGh/3tjfK1trZGR0dH2WNAJQgEStfR0eFOGaBinKQIACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACTzZnvDoigiImJkZOR/NgwAcHJNPm5PPo7PZNaBMDo6GhERXV1ds/0UAEBJRkdHo7W1dcbra8XxEmIGExMTsXfv3mhubo5arTbrAcsyMjISXV1dsWfPnmhpaSl7HL7A+lSfNao261NtZa5PURQxOjoaS5cujYaGmc80mPUzCA0NDbFs2bLZ3rwyWlpabJ4Ksz7VZ42qzfpUW1nr82XPHExykiIAkAgEACCp20BobGyMH/zgB9HY2Fj2KByD9ak+a1Rt1qfavgrrM+uTFAGAU1fdPoMAAMxMIAAAiUAAABKBAAAkp3wgbNu2LW644YZYunRp1Gq1ePHFF6ddf8cdd0StVpv21tvbW86wdWjTpk1x+eWXR3Nzc5xzzjnx7W9/O3bu3DntmLGxsdiwYUO0t7dHU1NT3HzzzXHgwIGSJq4vJ7I+V199ddpD99xzT0kT15ef/exnsXLlyqn/bGf16tXx6quvTl1v75TveGtU5f1zygfC4cOH49JLL42f/vSnMx7T29sb+/btm3r7zW9+M4cT1reBgYHYsGFDvPnmm/GXv/wljh49Gtdcc00cPnx46piHH344/vjHP8aWLVtiYGAg9u7dGzfddFOJU9ePE1mfiIi77rpr2h760Y9+VNLE9WXZsmXx1FNPxTvvvBNvv/12fPOb34wbb7wx/vGPf0SEvVMFx1ujiArvn6KORESxdevWaZf19fUVN954YynzkH300UdFRBQDAwNFURTFwYMHi/nz5xdbtmyZOuaf//xnERHF9u3byxqzbn1xfYqiKK666qriwQcfLG8opjn77LOLp59+2t6psMk1Kopq759T/hmEE9Hf3x/nnHNOXHDBBXHvvffGJ598UvZIdWt4eDgiItra2iIi4p133omjR4/Gt771raljLrzwwli+fHls3769lBnr2RfXZ9Kvf/3rWLRoUVx88cWxcePG+Oyzz8oYr679+9//jhdeeCEOHz4cq1evtncq6ItrNKmq+2fWP6zpVNHb2xs33XRTrFixInbt2hVPPPFErFu3LrZv3x6nnXZa2ePVlYmJiXjooYfiyiuvjIsvvjgiIvbv3x8LFiyIs846a9qxHR0dsX///hKmrF/HWp+IiFtuuSW6u7tj6dKlsWPHjnjsscdi586d8Yc//KHEaevHu+++G6tXr46xsbFoamqKrVu3xkUXXRSDg4P2TkXMtEYR1d4/dR8I3/3ud6fev+SSS2LlypVx3nnnRX9/f6xdu7bEyerPhg0b4u9//3v87W9/K3sUjmGm9bn77run3r/kkktiyZIlsXbt2ti1a1ecd955cz1m3bngggticHAwhoeH43e/+1309fXFwMBA2WPxX2Zao4suuqjS+8dLDF9w7rnnxqJFi+K9994re5S6cv/998crr7wSb7zxxrQfI97Z2Rmff/55HDx4cNrxBw4ciM7Ozjmesn7NtD7HcsUVV0RE2ENzZMGCBdHT0xOrVq2KTZs2xaWXXho//vGP7Z0KmWmNjqVK+0cgfMGHH34Yn3zySSxZsqTsUepCURRx//33x9atW+Ovf/1rrFixYtr1q1ativnz58frr78+ddnOnTtjaGho2mt4nBzHW59jGRwcjIiwh0oyMTER4+Pj9k6FTa7RsVRp/5zyLzEcOnRoWol98MEHMTg4GG1tbdHW1hZPPvlk3HzzzdHZ2Rm7du2K733ve9HT0xPXXnttiVPXjw0bNsTmzZvjpZdeiubm5qnXRltbW2PhwoXR2toad955ZzzyyCPR1tYWLS0t8cADD8Tq1avj61//esnTn/qOtz67du2KzZs3x3XXXRft7e2xY8eOePjhh2PNmjWxcuXKkqc/9W3cuDHWrVsXy5cvj9HR0di8eXP09/fHn/70J3unIr5sjSq/f8r+ZxQn2xtvvFFERHrr6+srPvvss+Kaa64pFi9eXMyfP7/o7u4u7rrrrmL//v1lj103jrU2EVE8++yzU8ccOXKkuO+++4qzzz67OOOMM4rvfOc7xb59+8obuo4cb32GhoaKNWvWFG1tbUVjY2PR09NTPProo8Xw8HC5g9eJ9evXF93d3cWCBQuKxYsXF2vXri3+/Oc/T11v75Tvy9ao6vvHj3sGABLnIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUCAOnHHHXdErVabemtvb4/e3t7YsWPH1DGT17355pvTbjs+Ph7t7e1Rq9Wiv79/2vEvvvjiHP0OgLkkEKCO9Pb2xr59+2Lfvn3x+uuvx7x58+L666+fdkxXV1c8++yz0y7bunVrNDU1zeWoQMkEAtSRxsbG6OzsjM7Ozrjsssvi8ccfjz179sTHH388dUxfX1+88MILceTIkanLnnnmmejr6ytjZKAkAgHq1KFDh+L555+Pnp6eaG9vn7p81apV8bWvfS1+//vfR0TE0NBQbNu2LW677bayRgVKIBCgjrzyyivR1NQUTU1N0dzcHC+//HL89re/jYaG6XcF69evj2eeeSYiIp577rm47rrrYvHixWWMDJREIEAd+cY3vhGDg4MxODgYb731Vlx77bWxbt262L1797Tjbr311ti+fXu8//778dxzz8X69etLmhgoi0CAOnLmmWdGT09P9PT0xOWXXx5PP/10HD58OH7xi19MO669vT2uv/76uPPOO2NsbCzWrVtX0sRAWQQC1LFarRYNDQ3TTkictH79+ujv74/bb789TjvttBKmA8o0r+wBgLkzPj4e+/fvj4iITz/9NH7yk5/EoUOH4oYbbkjH9vb2xscffxwtLS1zPSZQAQIB6shrr70WS5YsiYiI5ubmuPDCC2PLli1x9dVXp2NrtVosWrRojicEqqJWFEVR9hAAQLU4BwEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAg+Q8PIiSbI/3G7wAAAABJRU5ErkJggg==",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.boxplot(x=df['BMI'])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fVj81mFIuj2Q",
        "outputId": "4cdffd14-b8c5-4df6-a1c7-3ae6a122be6a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 9 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Sr.no                   1676 non-null   int64  \n",
            " 1   Name                    1676 non-null   object \n",
            " 2   Age(yrs)                1676 non-null   float64\n",
            " 3   Gender                  1676 non-null   object \n",
            " 4   Height (cm)             1676 non-null   float64\n",
            " 5   Weight(kg)              1676 non-null   float64\n",
            " 6   Health Insurance cover  1676 non-null   int64  \n",
            " 7   Premium                 1676 non-null   int64  \n",
            " 8   BMI                     1676 non-null   float64\n",
            "dtypes: float64(4), int64(3), object(2)\n",
            "memory usage: 118.0+ KB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "QALpMjtDuj5z",
        "outputId": "d0f1d41d-74f4-40c1-8c17-abb2d19400f6"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1676,\n  \"fields\": [\n    {\n      \"column\": \"Age(yrs)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 26.338582349667547,\n        \"min\": 0.25,\n        \"max\": 85.0,\n        \"num_unique_values\": 94,\n        \"samples\": [\n          34.0,\n          16.0,\n          49.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Health Insurance cover\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3170480,\n        \"min\": 500000,\n        \"max\": 10000000,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          7500000,\n          750000\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Premium\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 25308,\n        \"min\": 7015,\n        \"max\": 111340,\n        \"num_unique_values\": 82,\n        \"samples\": [\n          26410,\n          7015\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"BMI\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4.857805256764994,\n        \"min\": 13.770213967940117,\n        \"max\": 35.37981269510926,\n        \"num_unique_values\": 352,\n        \"samples\": [\n          24.99281815570238,\n          27.281746031746035\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "df"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-83cf5dca-b57f-4bf9-a80a-560992a70fcc\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age(yrs)</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <th>Premium</th>\n",
              "      <th>BMI</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.250000</td>\n",
              "      <td>male</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>16.976307</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.333333</td>\n",
              "      <td>male</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.143375</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.416667</td>\n",
              "      <td>male</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.269924</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.500000</td>\n",
              "      <td>male</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.287560</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.583333</td>\n",
              "      <td>male</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.332687</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1671</th>\n",
              "      <td>81.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1672</th>\n",
              "      <td>82.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1673</th>\n",
              "      <td>83.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1674</th>\n",
              "      <td>84.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1675</th>\n",
              "      <td>85.000000</td>\n",
              "      <td>female</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1676 rows × 5 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-83cf5dca-b57f-4bf9-a80a-560992a70fcc')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-83cf5dca-b57f-4bf9-a80a-560992a70fcc button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-83cf5dca-b57f-4bf9-a80a-560992a70fcc');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "  <div id=\"id_0ba225e7-db23-4c35-84fe-4cd73f70d547\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('df')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_0ba225e7-db23-4c35-84fe-4cd73f70d547 button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('df');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "       Age(yrs)  Gender  Health Insurance cover  Premium        BMI\n",
              "0      0.250000    male                  500000     7015  16.976307\n",
              "1      0.333333    male                  500000     7015  17.143375\n",
              "2      0.416667    male                  500000     7015  17.269924\n",
              "3      0.500000    male                  500000     7015  17.287560\n",
              "4      0.583333    male                  500000     7015  17.332687\n",
              "...         ...     ...                     ...      ...        ...\n",
              "1671  81.000000  female                10000000   111340  27.055151\n",
              "1672  82.000000  female                10000000   111340  27.055151\n",
              "1673  83.000000  female                10000000   111340  27.055151\n",
              "1674  84.000000  female                10000000   111340  27.055151\n",
              "1675  85.000000  female                10000000   111340  27.055151\n",
              "\n",
              "[1676 rows x 5 columns]"
            ]
          },
          "execution_count": 92,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.drop(['Sr.no', 'Name', 'Height (cm)', 'Weight(kg)'], axis=1, inplace=True)\n",
        "\n",
        "df\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0mo7LQw1EnX9"
      },
      "source": [
        "5. Feature Selection:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 599
        },
        "id": "tYrktXzfukAd",
        "outputId": "d473caff-4365-44fa-c9c5-df171ffb1eac"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: >"
            ]
          },
          "execution_count": 42,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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+Rx2drItL+ro8ZMQjHSdJX29cmPerdi6d3LE0nkzVBWWSt0/cROG5sboMq0fjKxM8TIoApuHLZdO6DEvZwGJ5P1QqpaD20NbqI3R22Sv16l1h3Fq3R/S1w2cm0EgEODusYJvo6Nujhw5AUsrc8ybPw0ikRUePQpCjx7DpJ/t8v/5bN++fR8jhk/BAvcZWLjoR4SFRmBA/7EIDHzfo9alS3v8+ttq6evde/IaOJYuXYdlS9chIzMTzZo3govLCJiamSA2Nh7Xr/mhXdve+SbsUckkkHzNMx2oVBLq2qo6BCqFUl5cVnUIam1aQ7dPFyK5dsWq943YVSk1LULpx7hRprfC6mr2+ojC6lIWpbRYEhEREREn7xARERERfRa2WBIREREpSWm7ay4TSyIiIiIlkYBd4URERERERcYWSyIiIiIlEZeye+8wsSQiIiJSEjG7womIiIiIio4tlkRERERKUtom7zCxJCIiIlIS3m6IiIiIiBSitLVYcowlERERESkEWyyJiIiIlIRd4URERESkEKUtsWRXOBEREREpBFssiYiIiJSktE3eYWJJREREpCTi0pVXsiuciIiIiBSDLZZERERESlLanhXOxJKIiIhISSSqDqCYsSuciIiIiBSCLZZERERESlLa7mPJxJKIiIhIScQCjrEkIiIiIgXgGEsiIiIios/AxJKIiIhIScQKXIpq8+bNqFixInR1dWFvbw8/P79C7XfgwAEIBAL06NGjyMdkYklERESkJGKB4paiOHjwIKZPnw53d3fcv38fderUgZOTE2JjYz+6X0REBGbOnImWLVt+1vkysSQiIiL6yqxduxZjxozBiBEjUKNGDWzbtg36+vrYuXNngfvk5uZi0KBBWLRoESpVqvRZx2ViSURERKQkYggUtmRmZiIpKUlmyczMzHfMrKws3Lt3D46OjtJ1GhoacHR0xM2bNwuMdfHixbC2tsaoUaM++3yZWBIREREpiUSBy/Lly2FiYiKzLF++PN8x4+PjkZubC5FIJLNeJBIhOjpabpzXrl2Dp6cntm/f/kXny9sNEREREakBNzc3TJ8+XWadUCj84nqTk5MxZMgQbN++HZaWll9UFxNLIiIiIiUp6qSbjxEKhYVKJC0tLaGpqYmYmBiZ9TExMbCxsclXPiwsDBEREXB2dpauE4vz5qFraWkhODgYlStXLlSMTCzpq5P46yBVh6C2TMbuVXUIamtaQzdVh6DWfrmbvzuPCiez4SxVh0AfoYpHOuro6KBBgwbw8fGR3jJILBbDx8cHkyZNyle+evXqCAgIkFk3b948JCcnY/369bC1tS30sZlYEhEREX1lpk+fjmHDhqFhw4Zo3Lgx1q1bh9TUVIwYMQIAMHToUJQrVw7Lly+Hrq4uatasKbO/qakpAORb/ylMLImIiIiURFWPdOzfvz/i4uKwYMECREdHo27dujhz5ox0Qk9UVBQ0NBQ/h5uJJREREZGSKHKMZVFNmjRJbtc3APj6+n50Xy8vr886JhNLIiIiIiVRxRhLVeJ9LImIiIhIIdhiSURERKQkpa3FkoklERERkZJIVDjGUhXYFU5ERERECsEWSyIiIiIlYVc4ERERESlEaUss2RVORERERArBFksiIiIiJVHVk3dUhYklERERkZKo8sk7qsCucCIiIiJSCLZYEhERESlJaZu8w8SSiIiISEmYWBIRERGRQpS2yTscY0lERERECsEWSyIiIiIlKW2zwplYEhERESlJaRtjya5wIiIiIlIItlgSERERKUlpm7zDxJKIiIhIScSlLLVkVzgRERERKQRbLImIiIiUpLRN3mFiSURERKQkpasjnF3hRERERKQgbLEkIiIiUhJ2hRMRERGRQpS2J++wK5y+SEREBAQCAfz9/VUdChERUYkjhkRhizpgi6WaGj58OLy9vQEA2traqFChAoYOHYo5c+ZAS6v4fqy2trZ4/fo1LC0ti+2YJdmBu8/gfTsEb1IyUE1kglkdaqNWWfMCy+/1C8Xh++GITkqDqZ4QjtXLYrLD9xBqaRZj1MrjvmAmRo4cCFNTE9y4eQeurnMQGhr+0X3Gjx+G6dPGw8bGCo8eBWHqtPm4e9dful0oFGLlyvno17c7hEIdnD9/Ga6T5yA2Nl5aZu3axWjWtCG+/94OT56EolFjp3zH6dO7K2bNckXVqpUQF/cGW7d5Ye3abQo7d2VrNaQD2o1zhrGVKV4GReKw+y5EPgyTW7bZgLZo3KsVytrZAgCiAsJxfNV+mfKbIg7K3fevZXvh89txxZ9ACXfXPwC79v2BwCehiHvzFuuXz0e7Vs1UHZbKtRniBKdx3WBiZYrnQZHY774TEQ9D5ZZtOaAdmvZqLX3fRQY8w1+r9suUH7HaBc36tJHZ7/Flf6wftlRp50DKxRZLNdaxY0e8fv0aISEhmDFjBhYuXIhVq1blK5eVlaW0GDQ1NWFjY1OsyWxJdTbwBdb4BGBci+rYP9IB1axNMPHADbxNzZRb/tQ/z7Hh0j8Y17I6/hzrCPcu9XAu6CU2+v5TzJErx8wZE+HiMgKTXN3QooUz0lLTcOLEXgiFwgL36dvHGatWLsDPS3+BvX0nPAoIxMkTe2FlZSEts3q1O7p0bo+BP4xDO8c+KFNGhEMHt+ery8v7IA4flp8QOTk5wNt7I37bvgf16rfD5ClzMNl1NCZMGP7F510c6ndtip7zhuL0+iNY0WU2XgZGwmX3HBhaGMstX7XJ97h37AbWD1yMNb3mI/H1G7jsmQsTkZm0jFujsTLL3h+3QiwWw//07eI6rRIlPT0DdlUqYe6MiaoOpcRo2LUZ+s0bhuPrD2NJl1l4ERiJqbvnwqiA951dk+/hd+wa1gxcBI9ec5Hw+g2m7ZkHU5Hsl+0A3weY0WiMdNnuuq4Yzqb4SBS4qAMmlmpMKBTCxsYG33zzDSZMmABHR0ccO3YMw4cPR48ePbB06VKULVsWdnZ2AIDnz5+jX79+MDU1hbm5Obp3746IiAhpff/ut2zZMohEIpiammLx4sXIycnBjz/+CHNzc5QvXx67du2S7vPfrnAvLy+YmprKxHn06FEIBO8HmSxcuBB169bFzp07UaFCBRgaGmLixInIzc3FypUrYWNjA2trayxdql7fWPf4haJX3YroUecbVLYyxrxOdaGrpYmjDyPkln/44i3qlrdA5+9tUc7UAM0qidCxRnk8fpVQvIEriavrKCz32IDjx88h4HEQRoycirJlROjeLX/r4b+mTBkLz537sXv3IQQ9CYGLy2ykpWVg+LABAABjYyOMGD4AP/20GL6+N/DgQQDGjJ2OZs0aoXHj+tJ6pk9fgG3bvBEeHiX3OIN+6I1jx85i+/a9CA+PwunTF7Fy1WbMVJMkou3oLrhxwAe3DvsiOvQlDszdgaz0LDTt5yC3vPfUjbi69xxeBkYiJuwVfp+1DQKBAHbNa0nLJMe9k1lqtW+IkJv/4M3z2OI6rRKlZdNGmDx2GBxbN1d1KCVG+9FdcfWAD24c9sXr0BfYO/c3ZKVnoXm/tnLL75i6Ab57z+F5YASiw17B+//vu++a15Qpl5OVjaS4ROmSlpRaHKdTbMQKXNQBE8uviJ6enrR10sfHB8HBwTh//jxOnDiB7OxsODk5wcjICFevXsX169dhaGiIjh07yrRoXrx4Ea9evcKVK1ewdu1auLu7o2vXrjAzM8Pt27cxfvx4jBs3Di9evPiiWMPCwnD69GmcOXMG+/fvh6enJ7p06YIXL17g8uXLWLFiBebNm4fbt9WjtSQ7V4yg14mwr2glXachEMD+Wys8evlW7j51ypsjMDoRAa/ytr9ISMW1sGi0qGxTLDEr07ffVkCZMiJc9LkqXZeUlAw/P3/YN2kgdx9tbW3Ur18LFy++30cikeDixato0iQvaaxfvxZ0dHTg80GZ4OAwREa+kJYpDKFQBxkZsi3J6ekZsLUti2++KV/oelRBU1sTtjUrIfh6gHSdRCJB8PUAfFu/aqHq0NETQlNbC2mJKXK3G1maoKZDPdw8eEkhMZP609TWwjc1KyHo+iPpOolEgqDrj1C5frVC1aGjpwNNbS2k/ud9Z9fke6y5uwNLfNZj0M9jYGBqqNDYqXix//IrIJFI4OPjg7Nnz8LV1RVxcXEwMDDAjh07oKOjAwDYu3cvxGIxduzYIW093LVrF0xNTeHr64sOHToAAMzNzbFhwwZoaGjAzs4OK1euRFpaGubMmQMAcHNzg4eHB65du4YBAwZ8dsxisRg7d+6EkZERatSoAQcHBwQHB+PUqVPSY69YsQKXLl2Cvb39F14h5UtIy0SuRAILA9luXgsDXUS8kf/Hu/P3tkhMy8KI3VcAADliCfrW+xajm9spPV5lE4nyEuyYD8Y9AkBsbBxsRFbydoGlpTm0tLQQExP3n33iYWdXBQBgI7JGZmYm3r1LylfGRmRd6PjOnb+M1avcsXvPIfj63kCVKhUxberYvGPYWCMy8su+OCmToZkxNLU0kRz/TmZ9Utw7iCqXLVQd3WcPwruYt3jyQXL6IfverZGRmgH/s35fHC99HQzNjKCppYkkOe87m8rlClVH79mDkRjzFoEfvO8eX36A+2duI/55LKy+EaHnjz9gitdcLO81FxKxurTRfZy6TLpRFCaWauzEiRMwNDREdnY2xGIxfvjhByxcuBAuLi6oVauWNKkEgIcPHyI0NBRGRkYydWRkZCAs7P0A/u+//x4aGu8bskUiEWrWfN9toampCQsLC8TGfln3WMWKFWViEYlE0NTUzHfsTx0nMzMTmZmyLU/i7BwItUv+W/tOZBw8bwRjTse6qFXWDM8TUrHy/CP8du0JxraorurwimTggJ7YvNlD+rp7j2EqjObTPD1/R+VK3+DoX97Q1tZCUlIKNm3yxIIFMyAWf91/BNpP6I4Gzs2wfsAi5GRmyy3TpF8b3D16rcDtREXVcUIPNHZujlUD3GXeV3eO35D++2VwFF4ERWL51c2wa1IDT248VkWoCvd1/0bJr+T/9aUCOTg4YOvWrdDR0UHZsmVlJtAYGBjIlE1JSUGDBg3w+++/56vHyup9C5K2trbMNoFAIHeduIBvkhoaGpBIZD9G2dn5/zh96XH+tXz5cixatEhm3ZzuzTGvZ4uP7qdoZvpCaAoEePOfiTpvUjNgaSB/ssqWy0HoUtMWvepWBABUtTZBenYOlpzyx+jmdtAQqM/Nz46fOAe/Ow+kr4X//1IjsrZEdPT7LwfW1lZ4+Ej+5KT4+LfIycmRtna+38cSMTF5dUTHxEIoFMLExFim1dLa2hLRMUX7sjNn7jLMm+8BGxtrxMW9Qdu2ee+Z8PDIItVT3FISkpCbkwsjSxOZ9cZWJkiKS/zovu3GdEX7Cd2xadDPePVE/vjTyo2qw6ZyOeyatF5RIdNXICUhGbk5uTD+jPddhzHO6DShB9YOWoyXBbzv/hX/PBbJb5JgXdHmq0ksSxuOsVRjBgYGqFKlCipUqPDJWdn169dHSEgIrK2tUaVKFZnFxMTko/sWhZWVFZKTk5Ga+n7wtTLvcenm5oZ3797JLD92baK04xVEW1MD35UxhV/E+25csUQCv4g41C4n/3ZDGTm5+ZLHf19L1OwrbkpKKsLCIqRLYNBTvH4dA4e27xN8IyNDNG5cF7dv3ZNbR3Z2Nu7fD4CDw/t9BAIBHBxa4Nat+wCA+/cDkJWVhbYflKlWrRK++aa8tExRiMVivHoVjezsbPTv1x03b95FfLz8MbElRW52Lp4/fga7Zu8n3ggEAlRrVhPh90MK3M9xXDd0dO2NLcOWIyrgWYHlmvZ3QNSjMLwMKtkJNhWv3OwcRD5+hu/+8777rlkthN1/WuB+TuO6oYtrH6wfthSRH3nf/cvMxhwGZoZ4F5uoiLBLhNI2eYctlqXEoEGDsGrVKnTv3h2LFy9G+fLlERkZiT///BM//fQTypdXzIQFe3t76OvrY86cOZg8eTJu374NLy8vhdQtj1AozHf7mnQVdYMPaVwF84/fQ40ypqhZ1gy/+4UhPTsX3Wt/AwCYd+wurI30MNnhewBAqyo22OsXiuoiU9QqZ4aohFRsuRKEVlVtoKmhPq2VBdm40RNusycjNDQcEeHPsXDhTLx6HYO/j52Vljlz5gD+/vsMtm71AgCsX/8bPD1/wf17D3Hnrj9cXUfDwEAP3rvz7rGYlJSMXV4HsHLlArxNSERSUjLW/bIEN2/ehZ/f+8SycuWKMDTQh8jGCnp6uqhTuwYAIDAoBNnZ2bCwMEOvXl1w5cpN6AqFGDqsP3r37op2jn2K7wJ9gYs7TmLImomICghDhH8YHEZ1hlBfiFuHfQEAQ9a44F3MWxxbuR8A4Di+G7pM6wfvKRvw5kUsjKzyvkxmpmYgK+19K7uuoR7qdW6Cv5buKfZzKmnS0tIR9eKV9PXLVzF48jQMJsZGKGNT+PG8X5PzO05g5BoXRASEIdw/FI6jukBHX4jrh/MmeY1cMwkJMW/x18p9AICO47uj27T+2DFlPeJfxMHYyhRA3vsuMy0DQn1dOE/pi/tnbuFdXCKsKojQx20I4iKi8c8VfxWdpeJxjCV9lfT19XHlyhXMmjULvXr1QnJyMsqVK4d27drB2Fj+Pcg+h7m5Ofbu3Ysff/wR27dvR7t27bBw4UKMHTtWYccoqZxqlEdCWia2XglCfGom7EQm2NK/GSwMdQEAr5PSZW67NKaFHQQCYPOVQMQmp8NMX4hWVWwwqU0NVZ2CQq1eswUGBvrYsnkFTE2Ncf3GHTg7D5YZE1vp229gafG+RffwH8dhaWWBBQtmwsbGCg8fBqKr8xCZm5/PnLkIYrEYBw/8JnOD9A9t27oKrVs3lb6+c+ccAKBqtSbSiTlDBvfFCo/5EAgEuHX7Hhzb95W5EXtJdv/ETRiaG6PLtH4wsjLFy6AIbB62XDqhx7ycBSSS9+0bLQe3h7ZQG6O3zZCp59S6wzi17g/p6wbOzSAQCHD32PXiOZES7PGTEIx0nSV9vXLjbwCA7p0csXTejIJ2+6rdPXEDRubG6D6tP4ytTPE8KALrhy394H1nKTMUqvXgDtAWamPCtpky9RxbdwjH1x2GOFeM8t9VQNPeraFvbIDE2LcIvPIIR9ceQE5WTrGeGymOQPLfAXFEai7de7aqQ1BbJmP3qjoEtTW2DO93+CV+ubtc1SGorYkNZ326EMm1PeKw0o8xreLn30Hlv36JOKCwupSFLZZERERESqIuYyMVhYklERERkZJIStkYS84KJyIiIiKFYIslERERkZKwK5yIiIiIFKK03W6IXeFEREREpBBssSQiIiJSktLVXsnEkoiIiEhp2BVORERERPQZ2GJJREREpCScFU5ERERECsEbpBMRERERfQa2WBIREREpCbvCiYiIiEghSltXOBNLIiIiIiUpbS2WHGNJRERERArBFksiIiIiJRFL2BVORERERApQutJKdoUTERERkYKwxZKIiIhISUrbs8KZWBIREREpSWm73RC7womIiIhIIdhiSURERKQkpe0+lkwsiYiIiJSktI2xZFc4ERER0Vdo8+bNqFixInR1dWFvbw8/P78Cy27fvh0tW7aEmZkZzMzM4Ojo+NHyBWFiSURERKQkEgX+VxQHDx7E9OnT4e7ujvv376NOnTpwcnJCbGys3PK+vr4YOHAgLl26hJs3b8LW1hYdOnTAy5cvi3RcJpZERERESiJW4JKZmYmkpCSZJTMzU+5x165dizFjxmDEiBGoUaMGtm3bBn19fezcuVNu+d9//x0TJ05E3bp1Ub16dezYsQNisRg+Pj5FOl8mlkRERERKIpFIFLYsX74cJiYmMsvy5cvzHTMrKwv37t2Do6OjdJ2GhgYcHR1x8+bNQsWdlpaG7OxsmJubF+l8OXmHiIiISA24ublh+vTpMuuEQmG+cvHx8cjNzYVIJJJZLxKJ8OTJk0Ida9asWShbtqxMcloYTCyJiIiIlESRs8KFQqHcRFLRPDw8cODAAfj6+kJXV7dI+zKxJCIiIlISVdzH0tLSEpqamoiJiZFZHxMTAxsbm4/uu3r1anh4eODChQuoXbt2kY/NxJK+OqkHCjd+hPJzEtVVdQhqa1fsbVWHoNYyG85SdQhqa8vdFaoOgUoYHR0dNGjQAD4+PujRowcASCfiTJo0qcD9Vq5ciaVLl+Ls2bNo2LDhZx2biSURERGRkqjqWeHTp0/HsGHD0LBhQzRu3Bjr1q1DamoqRowYAQAYOnQoypUrJ538s2LFCixYsAD79u1DxYoVER0dDQAwNDSEoaFhoY/LxJKIiIhISVT15J3+/fsjLi4OCxYsQHR0NOrWrYszZ85IJ/RERUVBQ+P9zYG2bt2KrKws9OnTR6Yed3d3LFy4sNDHZWJJRERE9BWaNGlSgV3fvr6+Mq8jIiIUckwmlkRERERKIpGUrmeFM7EkIiIiUhJVzApXJT55h4iIiIgUgi2WREREREqiqlnhqsLEkoiIiEhJVDUrXFWYWBIREREpSWmbvMMxlkRERESkEGyxJCIiIlISdoUTERERkUKUtsk77AonIiIiIoVgiyURERGRkohL2eQdJpZERERESlK60kp2hRMRERGRgrDFkoiIiEhJOCuciIiIiBSitCWW7AonIiIiIoVgiyURERGRkpS2RzoysSQiIiJSktLWFc7EkoiIiEhJ+OQdIiIiIqLPwBZLIiIiIiXhGEsiIiIiUojSNsaSXeFEREREpBBssSQiIiJSEnaFExEREZFCsCuciIiIiOgzsMWSiIiISElK230smVgSERERKYm4lI2xZFc4Kczw4cMhEAiki4WFBTp27IhHjx5Jy/y77datWzL7ZmZmwsLCAgKBAL6+vjLljx49WkxnUHS6XXvAzOsALP4+B5NftkKrWvWPlhcYGMJg4lSY//4nLI6dh9n2vdBuZC/drtdvEEzW/wrzI6dhvv8ojOb/DM1ytso+DZXoPLQLdlz3xJGnf2L132tQtU61AstWqFYBbtvcsOO6J45HnUC3Ud3yldEz0MNo9zHwvLETfzw9gpV/rkLV2lWVeQrFZuy4IQgMuoY3b4Phe/koGjSs89HyPXt2xv0HPnjzNhh+fmfg5NRGZnu37k44dmw3op4/QGpaBGrXrpGvjg0blyHg8WXEv3mCiMh7OHhoO6pVq6zI01KZNkOcsPzaZmwJ/h1uR5ehYp0qBZZtOaAdfjq0GOse7sK6h7swbe/8fOVHrHbB9ojDMssU77nKPo0S7a5/AFx+codDt0Go2bwTfK7cUHVIVEyYWJJCdezYEa9fv8br16/h4+MDLS0tdO3aVaaMra0tdu3aJbPur7/+gqGhYXGG+sV0WjnAYKwL0n73RqLrGOSGh8H459UQmJjK30FLC8bL1kBTZIOkpQuQMHoIkjesgjg+XlpEu1YdZBz/C++mTcC7OTMg0NKC8dLVgFC3eE6qmLRwbonR80dj/7r9mNplCsKDwrF472KYWJjILS/UFSI6KhreHt54G/tWbhnXla6o17Iu1k5dA9f2k/Dg6gMs2fczzEUWyjwVpevduys8POZh+bL1aN6sCwICAvH337thZSX/vOzt68PLewN2ex9Es6adcfzEORw4+Btq1HifuBvo6+PGzbuYP9+jwOM+eBCA8eN+RP16jujRfSgEAuDY8d3Q0FDvPxsNuzZDv3nDcHz9YSzpMgsvAiMxdfdcGFkYyy1v1+R7+B27hjUDF8Gj11wkvH6DaXvmwVRkLlMuwPcBZjQaI122u64rhrMpudLTM2BXpRLmzpio6lBUTqLA/9SBev+GoBJHKBTCxsYGNjY2qFu3LmbPno3nz58jLi5OWmbYsGE4cOAA0tPTpet27tyJYcOGqSLkz6bXsx8yTp9A5vnTyI2KRMrGNZBkZkC3Q2e55XU7dIaGkRGSFs9FTuBjiGOjkRPwELnhYdIySfN/QuaFM8iNikBueBiS1y6HpsgGWlULbs1TRz1G98DZ/Wfhc/gCnoc8xxa3zchMz0T7/u3llg95FIJdy3bh6vEryM7MzrddR6iDZp2aY9eyXfjH7x+8jnyN/b/sw+vI1+g8pJOyT0epXCePxq5dB7Bnz2E8eRKKya5zkZ6ejqFD+8ktP9FlJM6fv4x1635DcHAYlixeC3//fzBu/PvP1/79f8Fj+QZcuni9wOPu2rkf16/7ISrqBfz9/8HiRWtga1sO33xTXuHnWJzaj+6Kqwd8cOOwL16HvsDeub8hKz0Lzfu1lVt+x9QN8N17Ds8DIxAd9gres7ZBIBDgu+Y1ZcrlZGUjKS5RuqQlpRbH6ZRYLZs2wuSxw+DYurmqQ1E5sUSisEUdMLEkpUlJScHevXtRpUoVWFi8b11p0KABKlasiCNHjgAAoqKicOXKFQwZMkRVoRadlha0qlZDtv+99+skEmT734PWd9/L3UWnSXNkB/0DQ5dpMN/3F0y37oJe/8HAR1qABPp5rbiS5GSFhq9KWtpaqFKrCh5e85euk0gk8L/mD7v6Hx9KUBBNLU1oamki6z9JZ1ZGJmo0kv/zUAfa2tqoV68mLl16nwBKJBJcungdje3ry93H3r5evoTxwoUrsG8sv3xh6OvrYciQvggPj8KLF68/ux5V09TWwjc1KyHo+vvhORKJBEHXH6Fy/cJ9edPR04GmthZSE1Nk1ts1+R5r7u7AEp/1GPTzGBiYqlcPDCkPWyyJvsCJEydgaGgIQ0NDGBkZ4dixYzh48GC+7rORI0di586dAAAvLy907twZVlZWRT5eZmYmkpKSZJZMsVgh5/IxGsYmEGhqQZyQILNenJAADTNz+fvYlIGwRWtAQwPvFsxC2v7d0OvVD3oDCkioBQIYjpuE7H8eITcyXNGnoDLG5sbQ1NJEQnyizPrE+ESYWZl9Vp3pqekIuhuEAZMHwFxkDg0NDbTp2QZ29avDzPrz6iwJLCzNoKWlhdiYeJn1sbFxEInkf15EIivExsorb1nk448ZOxgxsf8gLj4I7Tu0gXPXwcjOzt9irC4MzYygqaWJpPh3MuuT4t7B2Mq0UHX0nj0YiTFvEXg9QLru8eUH2Dl9E9YOWowjK/aimn0NTPGaC4GaDxsg+hx815NCOTg4wN/fH/7+/vDz84OTkxM6deqEyMhImXKDBw/GzZs38ezZM3h5eWHkyJGfdbzly5fDxMREZlkfFqWIU1E4gUAD4sREpGxYjdzQp8i6cglpB/ZCr0t3ueUNXKZBs+K3SPZYXMyRqqe109ZAIAC87+zGn6F/wXlEN1z5+wokYvX4ll8SHTzwN5o17YIO7fshNPQZ9uzdDKFQqOqwVKbjhB5o7NwcW8atQs4HreN3jt/Awwt38TI4Cv7n7mDjyOX4tm4V2DXJPymKSh92hRN9AQMDA1SpUgVVqlRBo0aNsGPHDqSmpmL79u0y5SwsLNC1a1eMGjUKGRkZ6NTp88bBubm54d27dzLLlMoVFHEqHyVOegdJbg40zGRbwzTMzCBOkD+5RJzwBrkvnwMftKjmPo+EhrkFoCV75y+DCVOg07gp3s2aCnF83H+rUmtJb5OQm5MLM0tTmfWmlqZIiEuQv1MhREdGw62fG/rY9caIJsMxo9t0aGlrIjoq+gsjVp038QnIycmB9X9aG62trRATI/99ERMTB2treeXj5Zb/mKSkZISFReD6dT8M+mEiqlWrjG7dnIpcT0mRkpCM3JxcGFvKThIztjJBUlziR/ftMMYZnSb0wC9DluDlk49/eY1/HovkN0mwrmjzpSHTV4Bd4UQKJBAIoKGhITNR518jR46Er68vhg4dCk1Nzc+qXygUwtjYWGYRFkf3U04OckKeQrtug/frBAJo162PnKB/5O6S/c9jaJYtBwgE0nWa5coj9008kJMjXWcwYQp0mrXEu9lTIY5R36SoIDnZOQgNCEXt5u9vmSMQCFCneR0E33/yxfVnpmciITYBBiYGqNeqPm6fv/XpnUqo7OxsPHjwGG3aNJOuEwgEaOPQDH6378vd5/btB2jj0ExmXdu2LXDbT375wvr3VmFCoc4X1aNKudk5iHz8DN81qyVdJxAI8F2zWgi7/7TA/ZzGdUMX1z5YP2wpIgOeffI4ZjbmMDAzxLvYREWETaRWeIN0UqjMzExER+clQwkJCdi0aRNSUlLg7Oycr2zHjh0RFxcHY2P5t/ko6dL/OgSjGW7ICXmCnOAn0O3RBwKhHjLOnwYAGM6YA/GbOKR55bXWZpw8Ct1uPWEwfjLSjx2BZtny0O8/GOnHjkjrNHCZBmGbdkhaPBeS9HQI/j9eU5KaAmRlFf9JKsnRHUcxbc00hAaE4Kn/U3Qf1R26+rq4cOgCAGDaL9PxJvoNdq/wBpA34ce2at79PLV0tGAhssC3Nb5FRmoGXkfmTSap16o+BALg5bOXKFOxDEbMGYkXYS+kdaqrjRt24Lfta/DgfgDu3vWHy6RR0NfXx549hwEA27evwatXMXB3XwkA2LJ5J86eO4jJk0fjzJlL6NPXGfXr14LrJDdpnWZmJrC1LYcyZawBAFWrVgKQ19oZExOHihVt0aePMy74XEF83FuUK2eDGTMnID09A2fPXirmK6BY53ecwMg1LogICEO4fygcR3WBjr4Q1w/nndfINZOQEPMWf63cBwDoOL47uk3rjx1T1iP+RZx0LGZmagYy0zIg1NeF85S+uH/mFt7FJcKqggh93IYgLiIa/1zxV9FZql5aWjqiXrySvn75KgZPnobBxNgIZWysVRhZ8VOXLmxFYWJJCnXmzBmUKVMGAGBkZITq1avj8OHDaNOmTb6yAoEAlpZFn1BQUmRduYRUE1PoDx4JDXNz5ISFImn+j5Ak5nXnalpbA5L33d7i+Dgkzf0RBuNcYLZlJ8Rv4pH+9xGkH94nLaPXtQcAwHTlBpljJa9ZjswLZ5R/UsXk2vGrMDE3waDpg2FmZYZngc/gPmQBEv8/oceqrBUkHwwZMBeZY8OZjdLXvcb3Rq/xvRFwMwBz+uclTAbG+hg6axgsbSyR/C4ZN07dwJ5Vu5Gbk1us56ZoR46cgKWVOebNnwaRyAqPHgWhR49h0gk65W3LQfzBONLbt+9jxPApWOA+AwsX/Yiw0AgM6D8WgYHvW+S6dGmPX39bLX29e88mAMDSpeuwbOk6ZGRmolnzRnBxGQFTMxPExsbj+jU/tGvbG3Fxb4rpzJXj7okbMDI3Rvdp/WFsZYrnQRFYP2wpkv8/oce8nCUkHyQCrQd3gLZQGxO2zZSp59i6Qzi+7jDEuWKU/64CmvZuDX1jAyTGvkXglUc4uvYAcrJyUFo9fhKCka6zpK9XbvwNANC9kyOWzpuhqrBUQl26sBVFIJGUslSavnrxnVqrOgS1NeIfI1WHoLYuxj9WdQhq7QfrRqoOQW1tubtC1SGoLW3LSko/RiXLegqr61n8A4XVpSxssSQiIiJSEolE+bfAK0mYWBIREREpibiUdYVzVjgRERERKQRbLImIiIiUpLRNZWFiSURERKQkpa0rnIklERERkZKUthZLjrEkIiIiIoVgiyURERGRkvDJO0RERESkEKXtyTvsCiciIiIihWCLJREREZGSlLbJO0wsiYiIiJSktN1uiF3hRERERKQQbLEkIiIiUhJ2hRMRERGRQpS22w2xK5yIiIiIFIItlkRERERKwq5wIiIiIlKI0jYrnIklERERkZKUthZLjrEkIiIiIoVgiyURERGRkpS2WeFMLImIiIiURFLKxliyK5yIiIiIFIKJJREREZGSiCUShS1FtXnzZlSsWBG6urqwt7eHn5/fR8sfPnwY1atXh66uLmrVqoVTp04V+ZhMLImIiIiURCKRKGwpioMHD2L69Olwd3fH/fv3UadOHTg5OSE2NlZu+Rs3bmDgwIEYNWoUHjx4gB49eqBHjx54/PhxkY4rkJS2efD01Yvv1FrVIaitEf8YqToEtXUxvmi/fEnWD9aNVB2C2tpyd4WqQ1Bb2paVlH4MXd0KCqvr3bsQZGZmyqwTCoUQCoX5ytrb26NRo0bYtGkTAEAsFsPW1haurq6YPXt2vvL9+/dHamoqTpw4IV3XpEkT1K1bF9u2bSt0jGyxJCIiIlISiQL/W758OUxMTGSW5cuX5ztmVlYW7t27B0dHR+k6DQ0NODo64ubNm3LjvHnzpkx5AHByciqwfEE4K5yIiIhISRTZMezm5obp06fLrJPXWhkfH4/c3FyIRCKZ9SKRCE+ePJFbd3R0tNzy0dHRRYqRiSURERGRGiio27skYWJJREREpCSqmMpiaWkJTU1NxMTEyKyPiYmBjY2N3H1sbGyKVL4gHGNJREREpCQSBS6FpaOjgwYNGsDHx0e6TiwWw8fHB02bNpW7T9OmTWXKA8D58+cLLF8gCREVi4yMDIm7u7skIyND1aGoJV6/z8dr9/l47b4Mr5/qHDhwQCIUCiVeXl6SwMBAydixYyWmpqaS6OhoiUQikQwZMkQye/Zsafnr169LtLS0JKtXr5YEBQVJ3N3dJdra2pKAgIAiHZe3GyIqJklJSTAxMcG7d+9gbGys6nDUDq/f5+O1+3y8dl+G10+1Nm3ahFWrViE6Ohp169bFhg0bYG9vDwBo06YNKlasCC8vL2n5w4cPY968eYiIiEDVqlWxcuVKdO7cuUjHZGJJVEz4C/bL8Pp9Pl67z8dr92V4/UofjrEkIiIiIoVgYklERERECsHEkqiYCIVCuLu7l/h7kJVUvH6fj9fu8/HafRlev9KHYyyJiIiISCHYYklERERECsHEkoiIiIgUgoklERERESkEE0siIiIiUggmlkRERESkEEwsiajEkUgkiIqKQkZGhqpDUTvZ2dmoXLkygoKCVB0KEZVCTCyJqMSRSCSoUqUKnj9/rupQ1I62tjYTciJSGS1VB0D0NRKLxbh8+TKuXr2KyMhIpKWlwcrKCvXq1YOjoyNsbW1VHWKJpqGhgapVq+LNmzeoWrWqqsNROy4uLlixYgV27NgBLS3+mqficezYsUKV69atm5IjIVXiDdKJFCg9PR1r1qzB1q1b8fbtW9StWxdly5aFnp4e3r59i8ePH+PVq1fo0KEDFixYgCZNmqg65BLr+PHjWLlyJbZu3YqaNWuqOhy10rNnT/j4+MDQ0BC1atWCgYGBzPY///xTRZGplzt37uDSpUuIjY2FWCyW2bZ27VoVRVVyaWh8uhNUIBAgNze3GKIhVeFXWSIFqlatGpo2bYrt27ejffv20NbWzlcmMjIS+/btw4ABAzB37lyMGTNGBZGWfEOHDkVaWhrq1KkDHR0d6OnpyWx/+/atiiIr+UxNTdG7d29Vh6HWli1bhnnz5sHOzg4ikQgCgUC67cN/03v/Tb6pdGKLJZECBQUF4bvvvitU2ezsbERFRaFy5cpKjko9eXt7f3T7sGHDiikSKo1EIhFWrFiB4cOHqzoUIrXCxJKI6CuUk5MDX19fhIWF4YcffoCRkRFevXoFY2NjGBoaqjq8Eq9MmTK4cuUKx/gWwZUrVwpVrlWrVkqOhFSJiSWRkpw5cwaGhoZo0aIFAGDz5s3Yvn07atSogc2bN8PMzEzFEZZ8YWFh2LVrF8LCwrB+/XpYW1vj9OnTqFChAr7//ntVh1diRUZGomPHjoiKikJmZiaePn2KSpUqYcqUKcjMzMS2bdtUHWKJt3LlSrx69Qrr1q1TdShqQ0NDQzpMoKDUgmMsv3683RCRkvz4449ISkoCAAQEBGDGjBno3LkzwsPDMX36dBVHV/JdvnwZtWrVwu3bt/Hnn38iJSUFAPDw4UO4u7urOLqSbcqUKWjYsCESEhJkxqb+O6mHPm3mzJkIDg5G5cqV4ezsjF69eskslJ+ZmRlsbW0xf/58hISEICEhId/CsdFfPyaWREoSHh6OGjVqAACOHDmCrl27YtmyZdi8eTNOnz6t4uhKvtmzZ+Pnn3/G+fPnoaOjI13ftm1b3Lp1S4WRlXxXr17FvHnzZK4bAFSsWBEvX75UUVTqZfLkybh06RKqVasGCwsLmJiYyCyU3+vXr7FixQrcvHkTtWrVwqhRo3Djxg0YGxvz2pUinBVOpCQ6OjpIS0sDAFy4cAFDhw4FAJibm0tbMqlgAQEB2LdvX7711tbWiI+PV0FE6kMsFsvtbnzx4gWMjIxUEJH68fb2xpEjR9ClSxdVh6I2dHR00L9/f/Tv3x9RUVHw8vLCpEmTkJmZiWHDhmHRokW8r2opwBZLIiVp3rw5pk+fjiVLlsDPz0/6B+rp06coX768iqMr+UxNTfH69et86x88eIBy5cqpICL10aFDB5mxgQKBACkpKXB3d0fnzp1VF5gaMTc35x0bvkCFChWwYMECXLhwAdWqVYOHhwe/UJcSTCyJlGTz5s3Q1tbGH3/8ga1bt0qTodOnT6Njx44qjq7kGzBgAGbNmoXo6GgIBAKIxWJcv34dM2fOlLb+knxr1qzB9evXUaNGDWRkZOCHH36QdoOvWLFC1eGphYULF8Ld3V3a60CFl5mZiX379sHR0RE1a9aEpaUlTp48CXNzc1WHRsWAs8KJlCAnJwf79u1Dhw4dYGNjo+pw1FJWVhZcXFzg5eWF3NxcaGlpITc3Fz/88AO8vLygqamp6hBLtJycHBw4cACPHj1CSkoK6tevj0GDBuW70TzJV69ePYSFhUEikaBixYr5HnZw//59FUVWcvn5+WHXrl04cOAAKlasiBEjRmDw4MFMKEsZJpZESqKvr4+goCB88803qg5FrUVFReHx48dISUlBvXr1eF/BQsjIyICurq6qw1BrixYt+uh23pkgPw0NDVSoUAHDhg1DgwYNCizHZ4V/3ZhYEilJmzZtMHXqVPTo0UPVoVApY2xsjJ49e2Lw4MFo165doZ7hTPSl+KxwAjgrnEhpJk6ciBkzZuDFixdo0KABDAwMZLbXrl1bRZGVXEW5v+fatWuVGIl68/b2xr59+9C9e3eYmJigf//+GDx4MBo2bKjq0OgrxmeFE8AWSyKlkfftXSAQQCKR8Ft7ARwcHGRe379/Hzk5ObCzswOQN6NeU1MTDRo0wMWLF1URolpJTk7GH3/8gf379+PixYuoVKkSBg8ejAULFqg6tBLvw6fIyMPPb8HevHkDCwsLAMDz58+xfft2ZGRkwNnZGS1btlRxdKRsTCyJlCQyMvKj2zn28uPWrl0LX19feHt7Sx9/mZCQgBEjRqBly5aYMWOGiiNUL4GBgRg0aBAePXrEpKgQ/v77b5nX2dnZePDgAby9vbFo0SKMGjVKRZGVXAEBAXB2dsbz589RtWpVHDhwAB07dkRqaio0NDSQmpqKP/74g8ODvnJMLImUJDU1NV/3NxVeuXLlcO7cuXzPBH/8+DE6dOiAV69eqSgy9ZGRkYFjx45h3759OHPmDEQiEQYOHAgPDw9Vh6a29u3bh4MHD+ZLPAno1KkTtLS0MHv2bOzZswcnTpyAk5MTtm/fDgBwdXXFvXv3+OSsrxwTSyIlMTQ0RL9+/TBy5Ei0aNFC1eGoHSMjIxw/fhxt2rSRWX/p0iV069YNycnJqglMDZw9exb79u3D0aNHoaWlhT59+mDQoEFo1aqVqkNTe8+ePUPt2rWlz66n9ywtLXHx4kXp9TE2NsadO3ekM8SfPHmCJk2aIDExUbWBklJxqiCRkuzduxdv375F27ZtpU+eYCtb4fXs2RMjRozAn3/+iRcvXuDFixc4cuQIRo0ahV69eqk6vBKtZ8+eSE9Px+7duxEdHY1ff/2VSaUCpKenY8OGDXzyUwHevn0rvW+voaEhDAwMpMNYAMDMzIxfCEsBtlgSKVlcXBz27NkDLy8vBAUFwcnJCSNHjkS3bt343NyPSEtLw8yZM7Fz505kZ2cDALS0tDBq1CisWrWKwww+Ijk5mc8E/0JmZmYyk3ckEgmSk5Ohr6+PvXv38l6McmhoaCAmJgZWVlYA8nodHj16hG+//RYAEBMTg7Jly3KM71eOiSVRMdq4cSN+/PFHZGVlwdLSEuPHj8fs2bOhr6+v6tBKrNTUVISFhQEAKleuzISykHJzc3H06FEEBQUBAGrUqIHu3bvziUWF5OXlJZNYamhowMrKCvb29jKtcPSehoYGOnXqBKFQCAA4fvw42rZtK/3MZmZm4syZM0wsv3JMLImULCYmBt7e3vDy8kJkZCR69uyJUaNG4cWLF1ixYgXKli2Lc+fOqTpM+oqEhoaic+fOePnypfRWTcHBwbC1tcXJkydRuXJlFUdIX6MRI0YUqtyuXbuUHAmpEhNLIiX5888/sWvXLpw9exY1atTA6NGjMXjwYJiamkrLhIWF4bvvvkNWVpbqAi2hUlNT4eHhAR8fH8TGxua7+fKzZ89UFFnJ17lzZ0gkEvz+++/S5zS/efMGgwcPhoaGBk6ePKniCEumR48eoWbNmtDQ0MCjR48+WpYPOCCSj4klkZKYmJhgwIABGD16NBo1aiS3THp6OlauXMnnDssxcOBAXL58GUOGDEGZMmXy3ax6ypQpKoqs5DMwMMCtW7dQq1YtmfUPHz5E8+bNOaO5ABoaGoiOjoa1tbX0Buny/kTyAQdEBePMASIlef369SfHTurp6TGpLMDp06dx8uRJNG/eXNWhqB2hUCh39m1KSgp0dHRUEJF6CA8Pl048CQ8PV3E0ROqJiSWRAn14U/TCTMjhTdQLZmZmJu3GpaLp2rUrxo4dC09PTzRu3BgAcPv2bYwfP56zmT/iw6dh8clYRJ+HXeFEClSmTBlMmTIFw4YNQ5kyZeSWkUgkuHDhAtauXYtWrVrBzc2tmKNUD3v37sXff/8Nb29vzpovosTERAwbNgzHjx+HtrY2ACAnJwfdunWDl5cXTExMVByhenj16hWuXbsmd4zv5MmTVRQVUcnGxJJIgYKDgzFnzhycPHkSderUQcOGDVG2bFno6uoiISEBgYGBuHnzJrS0tODm5oZx48bx9i8FqFevHsLCwiCRSFCxYkVpgvSv+/fvqygy9REaGiq93dB3332HKlWqqDgi9eHl5YVx48ZBR0cHFhYWMmN8BQIBJ48RFYCJJZESREVF4fDhw7h69SoiIyORnp4OS0tL1KtXD05OTujUqRMTyk9YtGjRR7dzbCopk62tLcaPHw83NzdoaPAhdUSFxcSSiOgr07t3bzRu3BizZs2SWb9y5UrcuXMHhw8fVlFk6sPCwgJ+fn685ydREfFrGJGSZWVlITg4GDk5OaoORe0kJiZix44dcHNzw9u3bwHkdYG/fPlSxZGVbFeuXEHnzp3zre/UqROuXLmigojUz6hRo5iAE30GtlgSKUlaWhpcXV3h7e0NAHj69CkqVaoEV1dXlCtXDrNnz1ZxhCXbo0eP4OjoCBMTE0RERCA4OBiVKlXCvHnzEBUVhd27d6s6xBJLT08P/v7+0qfu/OvJkyeoV68e0tPTVRSZ+sjNzUXXrl2Rnp6OWrVq5Rvju3btWhVFRlSyscWSSEnc3Nzw8OFD+Pr6QldXV7re0dERBw8eVGFk6mH69OkYPnw4QkJCZK5f586d2er2CbVq1ZL7Hjtw4ABq1KihgojUz/Lly3H27FnExMQgICAADx48kC7+/v6qDo+oxOJ9LImU5OjRozh48CCaNGkiM6P0+++/R1hYmAojUw937tzBr7/+mm99uXLlEB0drYKI1Mf8+fPRq1cvhIWFoW3btgAAHx8f7N+/n927hbRmzRrs3LkTw4cPV3UoRGqFiSWRksTFxcHa2jrf+tTU1HyPJ6T8hEIhkpKS8q1/+vSp9OkoJJ+zszOOHj2KZcuW4Y8//oCenh5q166NCxcuoHXr1qoOTy0IhUI+9YnoM7ArnEhJGjZsiJMnT0pf/5tM7tixA02bNlVVWGqjW7duWLx4MbKzswHkXb+oqCjMmjULvXv3VnF0JV+XLl1w/fp1pKamIj4+HhcvXmRSWQRTpkzBxo0bVR0Gkdrh5B0iJbl27Ro6deqEwYMHS2+2HBgYiBs3buDy5cto0KCBqkMs0d69e4c+ffrg7t27SE5ORtmyZREdHY0mTZrg9OnTfBQmKVXPnj1x8eJFWFhY4Pvvv883eefPP/9UUWREJRsTSyIlCgsLg4eHBx4+fIiUlBTUr18fs2bNQq1atVQdmtq4fv26zPVzdHRUdUhUCowYMeKj23ft2lVMkRCpFyaWRFSipKenw8fHB127dgWQN7s+MzNTul1LSwuLFy+WmSlOREQlAyfvECmJvIknQN5YQaFQCB0dnWKOSD14e3vj5MmT0sRy06ZN+P7776Gnpwcg716MZcqUwbRp01QZJpUCOTk58PX1RVhYGH744QcYGRnh1atXMDY2hqGhoarDIyqR2GJJpCQaGhofnf1dvnx5DB8+HO7u7nwW8QdatmyJn376Cc7OzgAAIyMjPHz4EJUqVQIA7N27F5s3b8bNmzdVGaZayMrKQnh4OCpXrgwtLbYjFEVkZCQ6duyIqKgoZGZmSh9wMGXKFGRmZmLbtm2qDpGoROJfMyIl8fLyQtmyZTFnzhwcPXoUR48exZw5c1CuXDls3boVY8eOxYYNG+Dh4aHqUEuU0NBQmTGourq6Mol348aNERgYqIrQ1EZaWhpGjRoFfX19fP/994iKigIAuLq68v1WSFOmTEHDhg2RkJAgbS0H8ib1+Pj4qDAyopKNX2GJlMTb2xtr1qxBv379pOucnZ1Rq1Yt/Prrr/Dx8UGFChWwdOlSzJkzR4WRliyJiYkyYyrj4uJktovFYpntlN+HT33q2LGjdL2joyMWLlzIx4kWwtWrV3Hjxo18Q1YqVqzIZ9UTfQRbLImU5MaNG6hXr16+9fXq1ZN247Zo0ULamkR5ypcvj8ePHxe4/dGjRyhfvnwxRqR+jh49ik2bNqFFixZ86tNnEovFyM3Nzbf+xYsXMDIyUkFEROqBiSWRktja2sLT0zPfek9PT9ja2gIA3rx5AzMzs+IOrUTr3LkzFixYgIyMjHzb0tPTsWjRInTp0kUFkakPPvXpy3Xo0AHr1q2TvhYIBEhJSYG7uzs6d+6susCISjhO3iFSkmPHjqFv376oXr06GjVqBAC4e/cugoKCcOTIEXTt2hVbt25FSEgI1q5dq+JoS46YmBjUrVsXOjo6mDRpEqpVqwYACA4OxqZNm5CTk4MHDx5AJBKpONKSq1WrVujbty9cXV1hZGSER48e4dtvv4WrqytCQkJw5swZVYdY4j1//hwdO3aERCJBSEgIGjZsiJCQEFhaWuLKlStyE3ciYmJJpFQRERHYtm0bnj59CgCws7PDuHHjkJKSgpo1a6o4upIrPDwcEyZMwPnz5/HvryiBQID27dtjy5Yt0hniJB+f+qQYOTk5OHjwoMwN+gcNGiQzmYeIZDGxJComSUlJ2L9/P3bu3Im7d+/KHb9Fst6+fYvQ0FAAQJUqVWBubq7iiNQHn/r0+bKzs1G9enWcOHEC3333narDIVIrTCyJlOzKlSvw9PTEkSNHULZsWfTq1Qu9e/eWdo8TUclTrlw5XLhwgYklURFx8g6REkRHR8PDwwNVq1ZF3759YWxsjMzMTBw9ehQeHh5MKkmpTp06hbNnz+Zbf/bsWZw+fVoFEakfFxcXrFixAjk5OaoOhUitMLEkUjBnZ2fY2dnh0aNHWLduHV69eoWNGzeqOiwqRWbPni13qIVEIuE9LAvpzp07+PPP/7V3vzFVVw8cxz9fEFAokKbolQVmwywDt2BNydg0mRZbbTSsB1BqbqItw2qrVmvKA2u0KF0PCoVlNdLWqGClk3+6NZsbOViFkKgNKxoUksr17sq99/egn6yb/fr54AtHz/f92ti453wffB5+dr7nnG+DMjIytHLlShUXF0f9AfhnXJAOuGz//v3avHmzNm7cqKysLNNx4EEnTpzQHXfcccX4ggULxves4t9Nnz5dDz/8sOkYwHWHYgm47KuvvlJtba1yc3N1++23q6ysTI8++qjpWPCQlJQUnTp1SnPnzo0a7+vrU1JSkplQ14lwOKzXX39dP/zwg4LBoJYvX66tW7dyEhy4SrwKB1y2ePFi7dq1SwMDA9qwYYP27t2rOXPmKBwOq7m5WefPnzcdEZZ76KGHVFFREfWVnb6+Pj377LN68MEHDSa79l3+xOoNN9yg9PR07dy5U08++aTpWMB1g1PhwCTo7e1VbW2tPvjgA42MjKiwsFCNjY2mY8FSf/zxh1atWqWOjo7xz1/+9NNPuvfee9XQ0KDp06ebDXgNy8rK0nPPPacNGzZIklpaWlRUVKSLFy8qJoa1GOD/oVgCkygUCqmpqUl1dXUUS0yoSCSi5uZmdXV1adq0acrJyVFBQYHpWNe8hIQE9fX1jX92VZKmTp2qvr4+vlEPXAWKJQAA/xUbG6tff/1VM2fOHB/762cxAfw7Du8AgIVaW1vV2tqqwcFBhcPhqLm6ujpDqa59kUhEa9asUUJCwvhYIBBQeXl51MGnhoYGE/GAax7FEgAss23bNlVWViovL08+n0+O45iOdN14/PHHrxgrLS01kAS4PvEqHAAs4/P5VFVVpbKyMtNRAHgMR9wAwDLBYFD5+fmmYwDwIIolAFhm/fr1qq+vNx0DgAexxxIALBMIBFRTU6OWlhbl5OQoLi4uar66utpQMgC2Y48lAFhm2bJl/3POcRy1tbVNYhoAXkKxBAAAgCvYYwkAAABXsMcSACzU0dGhjz/+WP39/QoGg1FzXO4NYKKwYgkAltm7d6/y8/N1/Phxffrpp7p06ZK+//57tbW1KSUlxXQ8ABajWAKAZbZv364333xTTU1Nio+P144dO9TT06PVq1crIyPDdDwAFqNYAoBlTp48qaKiIklSfHy8RkdH5TiOtmzZopqaGsPpANiMYgkAlklNTdX58+clSenp6fruu+8kSSMjI/L7/SajAbAch3cAwDIFBQVqbm5Wdna2SkpK9PTTT6utrU3Nzc267777TMcDYDHusQQAywwPDysQCGjOnDkKh8OqqqrSkSNHlJWVpZdfflmpqammIwKwFMUSACwyNjam+vp6rVy5UrNmzTIdB4DHUCwBwDKJiYk6fvy4MjMzTUcB4DEc3gEAy9x9993q7Ow0HQOAB3F4BwAss2nTJj3zzDM6c+aMcnNzlZSUFDWfk5NjKBkA2/EqHAAsExNz5csox3EUiUTkOI5CoZCBVAC8gBVLALDM6dOnTUcA4FGsWAIAAMAVrFgCgGXef//9f51/7LHHJikJAK9hxRIALPP3C9AvXbokv9+v+Ph4JSYmanh42FAyALbjuiEAsMzZs2ej/i5cuKDe3l4tXbpUH330kel4ACzGiiUAeERHR4dKS0vV09NjOgoAS7FiCQAeMWXKFP3yyy+mYwCwGId3AMAyjY2NUb8jkYgGBgb09ttv65577jGUCoAX8CocACzz9wvSHcfRzJkztXz5cr3xxhvy+XyGkgGwHcUSAAAArmCPJQBYLhQKqbOzU2fPnjUdBYDlKJYAYJmKigrV1tZK+rNUFhQU6K677tLNN9+sQ4cOmQ0HwGoUSwCwzCeffKJFixZJkpqamvTjjz+qp6dHW7Zs0UsvvWQ4HQCbUSwBwDK//fabZs+eLUn68ssvVVJSovnz52vdunX69ttvDacDYDOKJQBYZtasWeru7lYoFNKBAwdUWFgoSfL7/YqNjTWcDoDNuMcSACyzdu1arV69Wj6fT47jaMWKFZKko0ePasGCBYbTAbAZxRIALLN161bdeeedOnPmjEpKSpSQkCBJio2N1QsvvGA4HQCbcY8lAAAAXMGKJQBYqLW1Va2trRocHFQ4HI6aq6urM5QKgO0olgBgmW3btqmyslJ5eXnj+ywBYDLwKhwALOPz+VRVVaWysjLTUQB4DNcNAYBlgsGg8vPzTccA4EEUSwCwzPr161VfX286BgAPYo8lAFgmEAiopqZGLS0tysnJUVxcXNR8dXW1oWQAbMceSwCwzLJly/7nnOM4amtrm8Q0ALyEYgkAAABXsMcSAAAArmCPJQBYori4+Kqea2homOAkALyKYgkAlkhJSTEdAYDHsccSAAAArmCPJQAAAFxBsQQAAIArKJYAAABwBcUSAAAArqBYAgAAwBVcNwQAFjpx4oTa29s1ODiocDgcNffKK68YSgXAdlw3BACW2bVrlzZu3KgZM2Zo9uzZchxnfM5xHB07dsxgOgA2o1gCgGUyMzO1adMmPf/886ajAPAYiiUAWCY5OVmdnZ2aN2+e6SgAPIbDOwBgmZKSEh08eNB0DAAexIolAFhg586d4/+Pjo6qurpaRUVFys7OVlxcXNSzmzdvnux4ADyCYgkAFrjllluu6jnHcXTq1KkJTgPAqyiWAAAAcAV7LAHAMpWVlfL7/VeMX7x4UZWVlQYSAfAKViwBwDKxsbEaGBhQWlpa1Pjvv/+utLQ0hUIhQ8kA2I4VSwCwTCQSiboU/bKuri7ddNNNBhIB8Ao+6QgAlkhNTZXjOHIcR/Pnz48ql6FQSBcuXFB5ebnBhABsx6twALDEnj17FIlEtG7dOr311ltKSUkZn4uPj9fcuXO1ZMkSgwkB2I5iCQCWOXz4sPLz86+4vxIAJhrFEgAscO7cuat+Njk5eQKTAPAyiiUAWCAmJuYfD+z81eVDPZwKBzBROLwDABZob283HQEAWLEEAACAO1ixBABL+f1+9ff3KxgMRo3n5OQYSgTAdhRLALDM0NCQ1q5dq/379//jPHssAUwUvrwDAJapqKjQyMiIjh49qmnTpunAgQPas2ePsrKy1NjYaDoeAIuxYgkAlmlra9Pnn3+uvLw8xcTEKDMzU4WFhUpOTtarr76qoqIi0xEBWIoVSwCwzOjoqNLS0iT9+ZnHoaEhSVJ2draOHTtmMhoAy1EsAcAyt912m3p7eyVJixYt0rvvvquff/5Z77zzjnw+n+F0AGzGdUMAYJkPP/xQY2NjWrNmjb755hutWrVKw8PDio+P13vvvadHHnnEdEQAlqJYAoDl/H6/enp6lJGRoRkzZpiOA8BiFEsAsFQwGNTp06d16623asoUzmoCmHjssQQAy/j9fj3xxBNKTEzUwoUL1d/fL0l66qmn9NprrxlOB8BmFEsAsMyLL76orq4uHTp0SFOnTh0fX7Fihfbt22cwGQDb8W4EACzz2Wefad++fVq8eLEcxxkfX7hwoU6ePGkwGQDbsWIJAJYZGhoav8fyr0ZHR6OKJgC4jWIJAJbJy8vTF198Mf77cpncvXu3lixZYioWAA/gVTgAWGb79u26//771d3drbGxMe3YsUPd3d06cuSIDh8+bDoeAIuxYgkAllm6dKk6Ozs1Njam7OxsHTx4UGlpafr666+Vm5trOh4Ai3GPJQAAAFzBq3AAsMS5c+eu6rnk5OQJTgLAq1ixBABLxMTE/Oup70gkIsdxFAqFJjEVAC9hxRIALNHe3j7+fyQS0QMPPKDdu3crPT3dYCoAXsKKJQBY6sYbb1RXV5fmzZtnOgoAj+BUOAAAAFxBsQQAAIArKJYAYDE+4QhgMnF4BwAsUVxcHPU7EAiovLxcSUlJUeMNDQ2TGQuAh1AsAcASKSkpUb9LS0sNJQHgVZwKBwAAgCvYYwkAAABXUCwBAADgCoolAAAAXEGxBAAAgCsolgAAAHAFxRIAAACuoFgCAADAFRRLAAAAuOI/Tuo9HCCC7y0AAAAASUVORK5CYII=",
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.heatmap(df.corr(), annot=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "N8G19fnXukDr",
        "outputId": "17d10f8d-4880-4daa-da5c-c36c500f4789"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.PairGrid at 0x7be8c0410230>"
            ]
          },
          "execution_count": 94,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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EVEOyIuB9tq1azPSbm1trehXxQHg3Xfflaz75JNPpj8gIiIiIiIiynu8ZUtERERERESURhxoExEREREREaURB9pEREREREREacSBNhEREREREVEacaBNRERERERElEZ5kXU81O7du/HOO+/g0KFD8Pl8YesWL16cpaiIiJJnsVhgtVoTLm8wGGA0GjMYERERERGlQ14NtJ944glcf/31KC0txcCBAyGTyYLrZDIZB9pElFVNTU0Jl+3o6MBpE05He1trwnWKikvw8daPUFhYmHCdXB2cJ/shA8C+pIIf5hAREWVHXg20lyxZgnvvvRfz5s3LdihEREFuhw2QyTFu3Lik606dvwb6fqVxy3Ue/hFv//EGnHDCCUm1X1zSD3t/2JNTgyeLxYLhx5ajrfVIUvXYl+SkElsuPsdERET5KK8G2q2trfjFL36R7TCIiMJ4XQ7A78PkeU+g0FSWUJ32H7/He4/cDFVBEXRF8QfaDsuRpLfhsBzBm0t+BavVmlMDJ6vVirbWIzjrjiehNfZLqA77kvnYcvU5JiIiykd5NdD+xS9+gTfffBO/+c1vsh0KEVEUbWG/hAbNQNfAOcPbyHVaI/vSE3I5NiIiot4q5wfaK1euDP5tNpuxaNEifPTRR6isrIRKpQore+ONN/Z0eERERERERERhcn6g/dBDD4X9v6CgAO+99x7ee++9sOUymYwDbSIiIiIiIsq6nB9o79mzJ9shEBERERERESVMnu0AknHPPffAZrNFLbfb7bjnnnuyEBERERERERFRuLwaaN99993o7OyMWm6z2XD33XdnISIiIiIiIiKicHk10Pb7/ZDJZFHLv/zyS/Trl9i0KkRERERERESZlPO/0QaAkpISyGQyyGQyHH/88WGDba/Xi87OTk75RURERERERDkhLwbaDz/8MPx+P66++mrcfffdKCoqCq5Tq9U49thjMXHixCxGSERERERERBSQFwPtWbNmAQDKy8txxhlnRM2fTURERERERJQrcn6gbbFYgn+PGzcOdrsddrtdtKzRaOypsIiIiIiIiIhE5fxAu7i4WDQBWighSZrX6+2hqDLjsMUBj9sLDwCr2wv4AT8Au8uDfgY1nF4f/P5AWRkC62QAfH7A5vKgn14Nly+8jNS6ROrnY5lciYN9Tr6M3w90Ojww6pQo1qtRZtSCiCgR+1tt6HB4YLG7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",
            "text/plain": [
              "<Figure size 1000x1000 with 20 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.pairplot(df)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LsuIfAntE4IR"
      },
      "source": [
        "6. model Building:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Yb0CqU4BukHX"
      },
      "outputs": [],
      "source": [
        "x = df.drop('Premium', axis=1)\n",
        "y = df['Premium']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Fj1WNlG7ukKo"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Gva1U-89Hfey"
      },
      "source": [
        "1. Linear Regression"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8ihv8zAmHtuH"
      },
      "source": [
        "#Model Evaluation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "osjmAaOJE2_M",
        "outputId": "6203ba0a-894c-4ce7-ffa8-dde9e94210f1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 103828434.46981017\n",
            "Root Mean Squared Error : 10189.623863019193\n",
            "Mean Absolute Error : 7900.433844436255\n",
            "R2 Score : 0.8324585751408807\n"
          ]
        }
      ],
      "source": [
        "# testing Accuracy:\n",
        "\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_test, y_pred)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "print(\"R2 Score :\",r2)\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6mdyJA4FE28B",
        "outputId": "551a1f9d-ee6f-4401-fdbb-f55c31443fde"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 115514230.08867213\n",
            "Root Mean Squared Error : 10747.75465335305\n",
            "Mean Absolute Error : 8150.092927382705\n",
            "R2 Score : 0.8219894882209786\n"
          ]
        }
      ],
      "source": [
        "#training accuracy:\n",
        "\n",
        "y_pred_train = linear_model.predict(x_train)\n",
        "\n",
        "mse = mean_squared_error(y_train, y_pred_train)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_train, y_pred_train)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_train, y_pred_train)\n",
        "print(\"R2 Score :\",r2)\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "d_8qUsM0I35v"
      },
      "source": [
        "2. Decision Tree:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "S_uGqLm1E248"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iyhisWbqOk_O"
      },
      "source": [
        "# Model Evaluation :"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ekuzJBlgE2vB",
        "outputId": "1e192a05-c021-4e20-bd13-150e2695a379"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 197624.25447316104\n",
            "Root Mean Squared Error : 444.5494960891993\n",
            "Mean Absolute Error : 28.031809145129223\n",
            "R2 Score : 0.9996811061502542\n"
          ]
        }
      ],
      "source": [
        "# testing accuracy :\n",
        "y_pred = dt_model.predict(x_test)\n",
        "\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_test, y_pred)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "print(\"R2 Score :\", r2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gT6Gto69OsCF",
        "outputId": "9068770b-d97e-4f97-8242-f99bd0999cbe"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 0.0\n",
            "Root Mean Squared Error : 0.0\n",
            "Mean Absolute Error : 0.0\n",
            "R2 Score : 1.0\n"
          ]
        }
      ],
      "source": [
        "#training accuracy :\n",
        "y_pred_train = dt_model.predict(x_train)\n",
        "\n",
        "mse = mean_squared_error(y_train, y_pred_train)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_train, y_pred_train)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_train, y_pred_train)\n",
        "print(\"R2 Score :\", r2)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "v0Mzn5dxO5Az"
      },
      "source": [
        "3. Random Forest :"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZpvIGoe0Or-Q"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 80
        },
        "id": "GuxXNculOr75",
        "outputId": "fa7bf872-d97e-4eb2-9e3f-c0529bac14fa"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<style>#sk-container-id-2 {\n",
              "  /* Definition of color scheme common for light and dark mode */\n",
              "  --sklearn-color-text: #000;\n",
              "  --sklearn-color-text-muted: #666;\n",
              "  --sklearn-color-line: gray;\n",
              "  /* Definition of color scheme for unfitted estimators */\n",
              "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
              "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
              "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
              "  --sklearn-color-unfitted-level-3: chocolate;\n",
              "  /* Definition of color scheme for fitted estimators */\n",
              "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
              "  --sklearn-color-fitted-level-1: #d4ebff;\n",
              "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
              "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
              "\n",
              "  /* Specific color for light theme */\n",
              "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
              "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-icon: #696969;\n",
              "\n",
              "  @media (prefers-color-scheme: dark) {\n",
              "    /* Redefinition of color scheme for dark theme */\n",
              "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
              "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-icon: #878787;\n",
              "  }\n",
              "}\n",
              "\n",
              "#sk-container-id-2 {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 pre {\n",
              "  padding: 0;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 input.sk-hidden--visually {\n",
              "  border: 0;\n",
              "  clip: rect(1px 1px 1px 1px);\n",
              "  clip: rect(1px, 1px, 1px, 1px);\n",
              "  height: 1px;\n",
              "  margin: -1px;\n",
              "  overflow: hidden;\n",
              "  padding: 0;\n",
              "  position: absolute;\n",
              "  width: 1px;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-dashed-wrapped {\n",
              "  border: 1px dashed var(--sklearn-color-line);\n",
              "  margin: 0 0.4em 0.5em 0.4em;\n",
              "  box-sizing: border-box;\n",
              "  padding-bottom: 0.4em;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-container {\n",
              "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
              "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
              "     so we also need the `!important` here to be able to override the\n",
              "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
              "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
              "  display: inline-block !important;\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-text-repr-fallback {\n",
              "  display: none;\n",
              "}\n",
              "\n",
              "div.sk-parallel-item,\n",
              "div.sk-serial,\n",
              "div.sk-item {\n",
              "  /* draw centered vertical line to link estimators */\n",
              "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
              "  background-size: 2px 100%;\n",
              "  background-repeat: no-repeat;\n",
              "  background-position: center center;\n",
              "}\n",
              "\n",
              "/* Parallel-specific style estimator block */\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel-item::after {\n",
              "  content: \"\";\n",
              "  width: 100%;\n",
              "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
              "  flex-grow: 1;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel {\n",
              "  display: flex;\n",
              "  align-items: stretch;\n",
              "  justify-content: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel-item {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel-item:first-child::after {\n",
              "  align-self: flex-end;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel-item:last-child::after {\n",
              "  align-self: flex-start;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-parallel-item:only-child::after {\n",
              "  width: 0;\n",
              "}\n",
              "\n",
              "/* Serial-specific style estimator block */\n",
              "\n",
              "#sk-container-id-2 div.sk-serial {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "  align-items: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  padding-right: 1em;\n",
              "  padding-left: 1em;\n",
              "}\n",
              "\n",
              "\n",
              "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
              "clickable and can be expanded/collapsed.\n",
              "- Pipeline and ColumnTransformer use this feature and define the default style\n",
              "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
              "*/\n",
              "\n",
              "/* Pipeline and ColumnTransformer style (default) */\n",
              "\n",
              "#sk-container-id-2 div.sk-toggleable {\n",
              "  /* Default theme specific background. It is overwritten whether we have a\n",
              "  specific estimator or a Pipeline/ColumnTransformer */\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "/* Toggleable label */\n",
              "#sk-container-id-2 label.sk-toggleable__label {\n",
              "  cursor: pointer;\n",
              "  display: flex;\n",
              "  width: 100%;\n",
              "  margin-bottom: 0;\n",
              "  padding: 0.5em;\n",
              "  box-sizing: border-box;\n",
              "  text-align: center;\n",
              "  align-items: start;\n",
              "  justify-content: space-between;\n",
              "  gap: 0.5em;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 label.sk-toggleable__label .caption {\n",
              "  font-size: 0.6rem;\n",
              "  font-weight: lighter;\n",
              "  color: var(--sklearn-color-text-muted);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 label.sk-toggleable__label-arrow:before {\n",
              "  /* Arrow on the left of the label */\n",
              "  content: \"▸\";\n",
              "  float: left;\n",
              "  margin-right: 0.25em;\n",
              "  color: var(--sklearn-color-icon);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "/* Toggleable content - dropdown */\n",
              "\n",
              "#sk-container-id-2 div.sk-toggleable__content {\n",
              "  max-height: 0;\n",
              "  max-width: 0;\n",
              "  overflow: hidden;\n",
              "  text-align: left;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-toggleable__content.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-toggleable__content pre {\n",
              "  margin: 0.2em;\n",
              "  border-radius: 0.25em;\n",
              "  color: var(--sklearn-color-text);\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-toggleable__content.fitted pre {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
              "  /* Expand drop-down */\n",
              "  max-height: 200px;\n",
              "  max-width: 100%;\n",
              "  overflow: auto;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
              "  content: \"▾\";\n",
              "}\n",
              "\n",
              "/* Pipeline/ColumnTransformer-specific style */\n",
              "\n",
              "#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator-specific style */\n",
              "\n",
              "/* Colorize estimator box */\n",
              "#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-label label.sk-toggleable__label,\n",
              "#sk-container-id-2 div.sk-label label {\n",
              "  /* The background is the default theme color */\n",
              "  color: var(--sklearn-color-text-on-default-background);\n",
              "}\n",
              "\n",
              "/* On hover, darken the color of the background */\n",
              "#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "/* Label box, darken color on hover, fitted */\n",
              "#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator label */\n",
              "\n",
              "#sk-container-id-2 div.sk-label label {\n",
              "  font-family: monospace;\n",
              "  font-weight: bold;\n",
              "  display: inline-block;\n",
              "  line-height: 1.2em;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-label-container {\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "/* Estimator-specific */\n",
              "#sk-container-id-2 div.sk-estimator {\n",
              "  font-family: monospace;\n",
              "  border: 1px dotted var(--sklearn-color-border-box);\n",
              "  border-radius: 0.25em;\n",
              "  box-sizing: border-box;\n",
              "  margin-bottom: 0.5em;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-estimator.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "/* on hover */\n",
              "#sk-container-id-2 div.sk-estimator:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-2 div.sk-estimator.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
              "\n",
              "/* Common style for \"i\" and \"?\" */\n",
              "\n",
              ".sk-estimator-doc-link,\n",
              "a:link.sk-estimator-doc-link,\n",
              "a:visited.sk-estimator-doc-link {\n",
              "  float: right;\n",
              "  font-size: smaller;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1em;\n",
              "  height: 1em;\n",
              "  width: 1em;\n",
              "  text-decoration: none !important;\n",
              "  margin-left: 0.5em;\n",
              "  text-align: center;\n",
              "  /* unfitted */\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted,\n",
              "a:link.sk-estimator-doc-link.fitted,\n",
              "a:visited.sk-estimator-doc-link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "/* Span, style for the box shown on hovering the info icon */\n",
              ".sk-estimator-doc-link span {\n",
              "  display: none;\n",
              "  z-index: 9999;\n",
              "  position: relative;\n",
              "  font-weight: normal;\n",
              "  right: .2ex;\n",
              "  padding: .5ex;\n",
              "  margin: .5ex;\n",
              "  width: min-content;\n",
              "  min-width: 20ex;\n",
              "  max-width: 50ex;\n",
              "  color: var(--sklearn-color-text);\n",
              "  box-shadow: 2pt 2pt 4pt #999;\n",
              "  /* unfitted */\n",
              "  background: var(--sklearn-color-unfitted-level-0);\n",
              "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted span {\n",
              "  /* fitted */\n",
              "  background: var(--sklearn-color-fitted-level-0);\n",
              "  border: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link:hover span {\n",
              "  display: block;\n",
              "}\n",
              "\n",
              "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
              "\n",
              "#sk-container-id-2 a.estimator_doc_link {\n",
              "  float: right;\n",
              "  font-size: 1rem;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1rem;\n",
              "  height: 1rem;\n",
              "  width: 1rem;\n",
              "  text-decoration: none;\n",
              "  /* unfitted */\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 a.estimator_doc_link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "#sk-container-id-2 a.estimator_doc_link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "#sk-container-id-2 a.estimator_doc_link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor(random_state=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>RandomForestRegressor</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html\">?<span>Documentation for RandomForestRegressor</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>RandomForestRegressor(random_state=10)</pre></div> </div></div></div></div>"
            ],
            "text/plain": [
              "RandomForestRegressor(random_state=10)"
            ]
          },
          "execution_count": 58,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "rf_model = RandomForestRegressor(random_state=10)\n",
        "rf_model.fit(x_train, y_train)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rixOCpmRPEm2"
      },
      "source": [
        "# Model Evaluation :"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wLYvzBl_Or55",
        "outputId": "f56107bf-6d1c-46ab-feda-decbf836be01"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 1910128.9930666\n",
            "Root Mean Squared Error : 1382.074163374238\n",
            "Mean Absolute Error : 234.4931411530814\n",
            "R2 Score : 0.996917744789303\n"
          ]
        }
      ],
      "source": [
        "# testing accuracy :\n",
        "y_pred = rf_model.predict(x_test)\n",
        "\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_test, y_pred)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "print(\"R2 Score :\", r2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W-2dKtrhOr3w",
        "outputId": "5826938f-8862-4857-bf74-1a9c30b3150e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mean Squared Error : 902738.5378623188\n",
            "Root Mean Squared Error : 950.1255379487064\n",
            "Mean Absolute Error : 131.46005967604424\n",
            "R2 Score : 0.9986088558179874\n"
          ]
        }
      ],
      "source": [
        " #training accuracy :\n",
        "y_pred_train = rf_model.predict(x_train)\n",
        "\n",
        "mse = mean_squared_error(y_train, y_pred_train)\n",
        "print(\"Mean Squared Error :\",mse)\n",
        "\n",
        "rmse = np.sqrt(mse)\n",
        "print(\"Root Mean Squared Error :\",rmse)\n",
        "\n",
        "mae = mean_absolute_error(y_train, y_pred_train)\n",
        "print(\"Mean Absolute Error :\",mae)\n",
        "\n",
        "r2 = r2_score(y_train, y_pred_train)\n",
        "print(\"R2 Score :\", r2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "3EFOnHvIOr1S",
        "outputId": "2958a0a2-f270-4d89-900b-bba3d6c298fd"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1676,\n  \"fields\": [\n    {\n      \"column\": \"Age(yrs)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 26.338582349667547,\n        \"min\": 0.25,\n        \"max\": 85.0,\n        \"num_unique_values\": 94,\n        \"samples\": [\n          34.0,\n          16.0,\n          49.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Health Insurance cover\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3170480,\n        \"min\": 500000,\n        \"max\": 10000000,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          7500000,\n          750000\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Premium\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 25308,\n        \"min\": 7015,\n        \"max\": 111340,\n        \"num_unique_values\": 82,\n        \"samples\": [\n          26410,\n          7015\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"BMI\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4.857805256764994,\n        \"min\": 13.770213967940117,\n        \"max\": 35.37981269510926,\n        \"num_unique_values\": 352,\n        \"samples\": [\n          24.99281815570238,\n          27.281746031746035\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "df"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-fbdb3f4d-0c17-498a-87b2-ea5361d07db8\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age(yrs)</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Health Insurance cover</th>\n",
              "      <th>Premium</th>\n",
              "      <th>BMI</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.250000</td>\n",
              "      <td>0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>16.976307</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.333333</td>\n",
              "      <td>0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.143375</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.416667</td>\n",
              "      <td>0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.269924</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.500000</td>\n",
              "      <td>0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.287560</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.583333</td>\n",
              "      <td>0</td>\n",
              "      <td>500000</td>\n",
              "      <td>7015</td>\n",
              "      <td>17.332687</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1671</th>\n",
              "      <td>81.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1672</th>\n",
              "      <td>82.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1673</th>\n",
              "      <td>83.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1674</th>\n",
              "      <td>84.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1675</th>\n",
              "      <td>85.000000</td>\n",
              "      <td>1</td>\n",
              "      <td>10000000</td>\n",
              "      <td>111340</td>\n",
              "      <td>27.055151</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1676 rows × 5 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fbdb3f4d-0c17-498a-87b2-ea5361d07db8')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-fbdb3f4d-0c17-498a-87b2-ea5361d07db8 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-fbdb3f4d-0c17-498a-87b2-ea5361d07db8');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "  <div id=\"id_70626992-3942-4310-9a2d-376296e82146\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('df')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_70626992-3942-4310-9a2d-376296e82146 button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('df');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "       Age(yrs)  Gender  Health Insurance cover  Premium        BMI\n",
              "0      0.250000       0                  500000     7015  16.976307\n",
              "1      0.333333       0                  500000     7015  17.143375\n",
              "2      0.416667       0                  500000     7015  17.269924\n",
              "3      0.500000       0                  500000     7015  17.287560\n",
              "4      0.583333       0                  500000     7015  17.332687\n",
              "...         ...     ...                     ...      ...        ...\n",
              "1671  81.000000       1                10000000   111340  27.055151\n",
              "1672  82.000000       1                10000000   111340  27.055151\n",
              "1673  83.000000       1                10000000   111340  27.055151\n",
              "1674  84.000000       1                10000000   111340  27.055151\n",
              "1675  85.000000       1                10000000   111340  27.055151\n",
              "\n",
              "[1676 rows x 5 columns]"
            ]
          },
          "execution_count": 62,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EsZvTeS8OrzN"
      },
      "outputs": [],
      "source": [
        "#Testing Model by taking data from user :\n",
        "\n",
        "age = 32\n",
        "gender = 'female'\n",
        "health_insurance_cover = 5000000\n",
        "height = 152\n",
        "weight = 51"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mpvDTWysOrw2",
        "outputId": "a85c6dc5-e87b-4fb6-9ee9-adc7e4af2c12"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "19.721036967980208"
            ]
          },
          "execution_count": 65,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "def gender_encoder(gender):\n",
        "  if gender=='male':\n",
        "    return 0\n",
        "  else:\n",
        "    return 1\n",
        "\n",
        "    gender_encoder('female')\n",
        "\n",
        "def bmi_converter(ht, wt):\n",
        "  bmi = wt/((ht/100)**2)\n",
        "  return bmi\n",
        "\n",
        "bmi_converter(167, 55)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fE0vq3f-OruZ",
        "outputId": "766f2dbc-4df8-4aa4-d86a-f6b343c69351"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1676 entries, 0 to 1675\n",
            "Data columns (total 5 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Age(yrs)                1676 non-null   float64\n",
            " 1   Gender                  1676 non-null   int64  \n",
            " 2   Health Insurance cover  1676 non-null   int64  \n",
            " 3   Premium                 1676 non-null   int64  \n",
            " 4   BMI                     1676 non-null   float64\n",
            "dtypes: float64(2), int64(3)\n",
            "memory usage: 65.6 KB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LfU_dBgtOrsA"
      },
      "outputs": [],
      "source": [
        "test_df = pd.DataFrame({'Age(yrs)':[age],\n",
        "                        'Gender':[gender_encoder(gender)],\n",
        "                        'Health Insurance cover':[health_insurance_cover],\n",
        "                        'BMI':[bmi_converter(height, weight)]})"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VjWXYhFVOrpi",
        "outputId": "bfcef843-3065-4478-9456-d084d1b47073"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([16795.])"
            ]
          },
          "execution_count": 68,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "test_df\n",
        "rf_model.predict(test_df)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Nv2WvgwLOrm0"
      },
      "outputs": [],
      "source": [
        " #pickling the model :\n",
        "import pickle\n",
        "file = open('rf_model_file.pkl', 'wb')\n",
        "pickle.dump(rf_model, file)\n",
        "file.close()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NRJvSd3nQUuT",
        "outputId": "cade3660-f4fa-4bc0-ad24-60fe1725860c"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([16795.])"
            ]
          },
          "execution_count": 72,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Unpicking the file :\n",
        "\n",
        "f = open('rf_model_file.pkl', 'rb')\n",
        "rf_new_model = pickle.load(f)\n",
        "f.close()\n",
        "\n",
        "rf_new_model.predict(test_df)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

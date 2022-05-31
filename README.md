# kaggle-spaceship-titanic
I made a simple NN for binary classification.

## Steps I did during making model:

Data preprocessing

Train data has following columns: ["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name", "Transported"].

First step was to drop columns, that has no impact on result, such as "PassengerId" and "Name". Then, Nan values were filled in such way:
1) Non-numeric columns were replaced with mode value;
2) Numeric columns were replaced with mean value.

After this, numeric values were scaled by MinMaxScaler. If we look at "Cabin" column, we will see, that it has mask L1/N/L2, where L means letter and N stands for number. To handle with this column it was chosen to separate L1 and L2 from it.

Cabin number was droped, but i will try to split numbers into intervals and one-hot them later.
Last step was to one-hot remaining columns. At this moment data preprocessing stage is over.

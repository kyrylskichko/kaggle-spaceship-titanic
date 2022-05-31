# kaggle-spaceship-titanic
I made a simple NN for binary classification.

## Steps I did during making model:

Data preprocessing

Train data has following columns: [PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name,Transported]
First step was to drop columns, that has no impact on result, such as PassengerId and Name. Then, Nan values were filled in such way:
1) Non-numeric values were replaced with mode value
2) Numeric values were replaced with mean value

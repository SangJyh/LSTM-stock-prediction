# container-for-stock-prediction
This is a Docker container built from scratch

## How to use it?

### Pull from Docker Repository and Run

```bash
docker pull even25/stock-lstm
docker run even25/stock-lstm python app.py --name `STOCK NAME`
#the output
`Predicted Price`
`STOCK NAME` historical stock price (from `Y-M-D` to `Y-M-D`)
Next: `Bull/Bear/Same`!
```


### Run Remotely

```bash
docker run -it even25/stock-lstm python app.py --name `STOCK NAME`
#the output
`Predicted Price`
`STOCK NAME` historical stock price (from `Y-M-D` to `Y-M-D`)
Next: `Bull/Bear/Same`!
```

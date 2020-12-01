window.onload = () => {
    main();
}

const main = () => {
    formSubmit();
}

const formSubmit = () => {
    const form = document.getElementById('fileUpload');
    form.addEventListener('submit', e => {
        e.preventDefault();
        const file = document.getElementById('file');
        const inputFile = file.files[0];
        const reader = new FileReader();
        reader.onload = function () {
            const text = reader.result;
            const json = tsv2Json(text);
            tfLR(json)
        };
        reader.readAsText(inputFile);
    })
}


const tsv2Json = (csv) => {
    const lines = csv.replace(/"/g, '').split(/[\r\n]+/g);
    const result = [];
    const headers = lines[0].replace(/"/g, '').split(/[,\t]/g);
    for (let i = 1; i < lines.length; i++) {
        const obj = {};
        const currentline = lines[i].split(/[,\t]/g);
        for (let j = 0; j < headers.length; j++) {
            if (headers[j] === 'PID') continue;
            if (currentline[j]) {
                let value = headers[j];
                obj[value] = parseInt(currentline[j]);
            }
        }
        if (Object.keys(obj).length > 0) result.push(obj);
    }
    headers.splice(headers.indexOf('PID'), 1);
    return {
        data: result,
        headers
    };
}

const logit = (x) => {
    return - Math.log(1.0 / x - 1.0)
}

const createDataset = (data, testSize, batchSize) => {
    const y = data.map(r => {
        const outcome = r['case.control'];
        return oneHot(outcome);
    });
    const X = data.map(dt => {
        delete dt['case.control']
        return tf.sigmoid(Object.values(dt)).dataSync().map(x => logit(x));
    });

    const splitIdx = parseInt((1 - testSize) * data.length, 10);

    const ds = tf.data
        .zip({
            xs: tf.data.array(X),
            ys: tf.data.array(y)
        })
        .shuffle(data.length, 42);

    return [
        ds.take(splitIdx).batch(batchSize),
        ds.skip(splitIdx + 1).batch(batchSize),
        tf.tensor(X.slice(splitIdx)),
        tf.tensor(y.slice(splitIdx))
    ];
}

const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());

const trainLogisticRegression = async (featureCount, trainDs, validDs, epochs) => {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 2,
            activation: "softmax",
            inputShape: [featureCount]
        })
    );
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    });
    
    const trainLogs = [];
    const lossContainer = document.getElementById("lossCont");
    const accContainer = document.getElementById("accCont");
    
    await model.fitDataset(trainDs, {
        epochs: epochs,
        validationData: validDs,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                trainLogs.push(logs);
                tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
                tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
            }
        }
    });
    return model;
};

const tfLR = async (json) => {
    const data = json.data.slice(0, 1000);
    const headers = json.headers;
    const [trainDs, validDs, xTest, yTest] = createDataset(data, 0.5, 50);
    
    const model = await trainLogisticRegression(
        headers.length-1,
        trainDs,
        validDs,
        10
    );
    model.summary();
    console.log(model.getWeights())
    for (let i = 0; i < model.getWeights().length; i++) {
        console.log(model.getWeights()[i].dataSync());
    }
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = document.getElementById("confusionMatrix")
    tfvis.render.confusionMatrix(container, {
        values: confusionMatrix,
        tickLabels: ["Control", "Case"],
    })
}
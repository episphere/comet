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
    const obj = {};
    for (let i = 1; i < lines.length; i++) {
        const currentline = lines[i].split(/[,\t]/g);
        for (let j = 0; j < headers.length; j++) {
            if (headers[j] === 'PID') continue;
            if(obj[headers[j]] === undefined) obj[headers[j]] = [];
            obj[headers[j]].push(parseInt(currentline[j]));
        }
    }
    let outcomes = [];
    for(let key in obj) {
        if(key === 'case.control') outcomes = obj[key]
        else result.push(obj[key])
    }
    headers.splice(headers.indexOf('PID'), 1);
    return {data: result, outcomes, headers};
}


const logit = (x) => {
    return - Math.log(1.0 / (1.0 - x))
}

const oneHot = data => Array.from(tf.oneHot(data, 4).dataSync());

const trainLogisticRegression = (data, headers, outcomes, epochs, batchSize) => {
    const X = data;
    const Y = outcomes;
    const weights = [];
    X.forEach(snip => {
        const model = tf.sequential();
        model.add(
            tf.layers.dense({
                units: 1,
                activation: "softmax",
                inputShape: [1]
            })
        );
        // model.summary();
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });
        // const result = model.evaluate(tf.tensor1d(snip), tf.tensor1d(Y), {batchSize: batchSize});
        // result.print();
        // model.predict(snip, {batchSize: batchSize}).print()

        model.fit(tf.tensor1d(snip), tf.tensor1d(Y), {
            batchSize: batchSize,
            epochs: epochs
        });
        weights.push(model.getWeights()[0].dataSync()[0])
    });

    let template = `<table><thead><tr><th>SNP_name</th><th>Estimate</th></tr></thead><tbody>`
    
    for(let i =0; i < headers.length; i++) {
        if(headers[i] !== 'case.control') {
            template += `<tr><td>${headers[i]}</td><td>${weights[i]}</td></tr>`
        }
    };

    `</tbody></table>`
    document.getElementById('output').innerHTML = template;
}

const tfLR = async (json) => {
    const data = json.data;
    const headers = json.headers;
    const outcomes = json.outcomes;
    trainLogisticRegression(data, headers, outcomes, 10, 500);
}
window.onload = () => {
    main();
}

const main = () => {
    formSubmit();
}

const formSubmit = () => {
    const form = document.getElementById('fileUpload');
    form.addEventListener('submit', async e => {
        e.preventDefault();
        const file = document.getElementById('file');
        const url = document.getElementById('url');
        const inputFile = file.files[0];
        if(url.value) {
            document.getElementById('output').innerHTML = 'Extracting results...';
            const response = await fetch(url.value);
            let json = {};
            if(url.value.includes('.json')) {
                const jsonData = await response.json()
                const species = ['setosa', 'versicolor', 'virginica'];
                species.forEach(type =>{
                    if(json.data === undefined) json.data = {};
                    if(json.outcomes === undefined) json.outcomes = {};
                    json.data[`${type}.petalLength`] = jsonData.map(dt => dt.petalLength)
                    json.data[`${type}.petalWidth`] = jsonData.map(dt => dt.petalWidth)
                    json.data[`${type}.sepalLength`] = jsonData.map(dt => dt.sepalLength)
                    json.data[`${type}.sepalWidth`] = jsonData.map(dt => dt.sepalWidth)
                    json.outcomes[type] = jsonData.map(dt => dt.species === type ? 1 : 0)
                })
                tfLRIrisds(json)
            }
            else{
                const text = await response.text();
                json = tsv2Json(text);
                trainLogisticRegression(json)
            }
        }
        else if(inputFile) {
            const reader = new FileReader();
            reader.onload = function () {
                document.getElementById('output').innerHTML = 'Extracting results...';
                const text = reader.result;
                const json = tsv2Json(text);
                trainLogisticRegression(json)
            };
            reader.readAsText(inputFile);
        }
    })
}

const tsv2Json = (csv) => {
    const lines = csv.replace(/"/g, '').split(/[\r\n]+/g);
    const result = [];
    const headers = lines[0].replace(/"/g, '').split(/[,\t]/g);
    let outcomeColumn = headers[headers.length-1];
    const obj = {};
    
    for (let i = 1; i < lines.length; i++) {
        if(!lines[i]) continue;
        const currentline = lines[i].split(/[,\t]/g);
        for (let j = 0; j < headers.length; j++) {
            if (headers[j] === 'PID') continue;
            if(obj[headers[j]] === undefined) obj[headers[j]] = [];
            obj[headers[j]].push(parseInt(currentline[j]));
        }
    }
    let outcomes = [];
    for(let key in obj) {
        if(key === outcomeColumn) outcomes = obj[key]
        else result.push(obj[key])
    }
    headers.splice(headers.indexOf('PID'), 1);
    return {data: result, outcomes, headers};
}

const trainLogisticRegression = (json) => {
    const data = json.data;
    const headers = json.headers;
    const outcomes = json.outcomes;
    const epochs = 10;
    const X = data;
    const Y = outcomes;
    const weights = [];
    const stdErr = [];
    X.forEach(snip => {
        const model = tf.sequential();
        model.add(
            tf.layers.dense({
                units: 1,
                activation: "softmax",
                inputShape: [1]
            })
        );
        model.compile({
            optimizer: tf.train.sgd(0.01),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });

        model.fit(tf.tensor1d(snip), tf.tensor1d(Y), {
            epochs: epochs
        });
        
        weights.push(model.getWeights()[0].dataSync()[0]);
        const standardDeviation = tf.moments(tf.tensor1d(snip)).variance.sqrt().dataSync()[0];
        stdErr.push(standardDeviation/(Math.sqrt(snip.length)))
    });

    let template = `<table><thead><tr><th>SNP_name</th><th>Weights</th><th>Standard error</th></tr></thead><tbody>`;

    for(let i =0; i < headers.length; i++) {
        if(headers[i] !== 'case.control') {
            template += `<tr><td>${headers[i]}</td><td>${weights[i]}</td><td>${stdErr[i]}</td></tr>`
        }
    };

    `</tbody></table>`
    document.getElementById('output').innerHTML = template;
}

const tfLRIrisds = (json) => {
    const data = json.data;
    const outcomes = json.outcomes;
    let predictions = {};
    const weights = {};
    const stdErr = {};
    for(let type in data) {
        const X = data[type];
        const Y = outcomes[type.substr(0, type.indexOf('.'))];
        
        const w0 = tf.scalar(Math.random()).variable();
        const w1 = tf.scalar(Math.random()).variable();
        
        const xs = tf.tensor1d(X);
        const ys = tf.tensor1d(Y);

        const f = x => tf.div(1, tf.add(1, ((w0.add(w1.mul(x))).neg()).exp()))

        const loss = (pred, label) => ((pred.sub(label)).square()).mean();
        
        const optimizer = tf.train.sgd(0.001);

        for (let i = 0; i < 500; i++) {
            optimizer.minimize(() => loss(f(xs), ys));
        }

        predictions[type] = f(xs).dataSync();

        const standardError = (pred, actual) => (((pred.sub(actual)).square()).sum()).sqrt();
        weights[type] = {};
        weights[type]['w0'] = w0.dataSync()[0].toFixed(5);
        weights[type]['w1'] = w1.dataSync()[0].toFixed(5);
        stdErr[type] = ((standardError(f(xs), ys).dataSync()[0])/X.length).toFixed(5);
    }
    let template = `<table><thead><tr><th></th><th>W<sub>0</sub></th><th>W<sub>1</sub></th><th>Standard error</th></tr></thead><tbody>`;

    for(let key in predictions) {
        template += `<tr><td>${key}</td><td>${weights[key].w0}</td><td>${weights[key].w1}</td><td>${stdErr[key]}</td></tr>`;
    }

    `</tbody></table>`
    console.log(predictions);
    
    document.getElementById('output').innerHTML = template;;
}

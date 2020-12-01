const tf = require('@tensorflow/tfjs');
const BoxSDK = require('box-node-sdk');
fs = require('fs');

const main = async () => {
    // const client = BoxSDK.getBasicClient('DDsT2tnXalubgvPnW8UVSsOCQzWCNZvJ');
    // await client.files.getReadStream(735183322567, null, (error, stream) => {
    //     console.log(stream)
    // })
    const content = fs.readFileSync('C:/Users/patelbhp/Desktop/PLCO_GSA_72_genotypes.txt', {encoding: 'utf8'});
    console.log(tsv2Json(content))
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

main();
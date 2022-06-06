function table_to_csv(source) {
    //const columns = Object.keys(source.data)
    var columns = source.columns() //
    const nrows = source.get_length()
    // const lines = [columns.join(',')]
    const ignore = ["LitScore", "CodonColor", "index","logPvalue","PubMed","Haplotype", "Position"]
    const lines = [columns.filter(function(str){return ignore.indexOf(str) < 0;}).join('\t')]
    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            if (ignore.indexOf(column) > -1 )
            {
                continue
            }

            if (column == "GeneName")
            {
                const s = source.data["GeneName"][i].toString();
                const start = s.indexOf(">");
                const end = s.indexOf("</");
                row.push(s.slice(start+1, end));
                continue;
            } 
            
            // if (column.startsWith("PMIDs_"))
            // {
            // }
            row.push(source.data[column][i].toString())
            
        }
        lines.push(row.join('\t'))
    }
    return lines.join('\n').concat('\n')
}


const filename = dataset + '.result.txt'
const filetext = table_to_csv(source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}

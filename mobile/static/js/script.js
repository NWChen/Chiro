$(document).ready(function() {
    var snapshot = document.getElementById('snapshot');
    var player = document.getElementById('player');
    var button = document.getElementById('capture');
    console.log("CLICK");
    // Handle successful webcam image sample
    var handleSuccess = function(stream) {
        player.srcObject = stream;
    }

    //button.addEventListener('click', function() {
    setInterval(function() {
        var context = snapshot.getContext('2d');
        context.drawImage(player, 0, 0, snapshot.width, snapshot.height);
        var dataURL = snapshot.toDataURL("image/jpeg");
        console.log(dataURL); 

        var img = new Image();
        img.src = dataURL;

        var blobBin = atob(dataURL.split(',')[1]);
        var array = [];
        for(var i = 0; i < blobBin.length; i++) {
            array.push(blobBin.charCodeAt(i));
        }
        var file = new Blob([new Uint8Array(array)], {type: 'image/png'});

        var formdata = new FormData();
        formdata.append("file", file);
        $.ajax({
            url: "/upload",
            type: "POST",
            data: formdata,
            processData: false,
            contentType: false,
        }).done(function(respond){
            alert(respond);
        });
        //document.getElementById('image-file').value = img; 
        //document.getElementById('image-form').submit();
        /* 
        $.ajax({
            type: "POST",
            url: "/upload",
            data: {
                imgBase64: dataURL
            }
        }).done(function(o) {
            console.log("SAVED");
        });
        */
        //var image = dataURL.replace("image/png", "image/octet-stream");
        //var download = document.getElementById("download");
        //download.setAttribute("href", image);
    }, 2000);

    navigator.mediaDevices.getUserMedia({video: true}).then(handleSuccess);
});

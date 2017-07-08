$(document).ready(function() {
    var snapshot = document.getElementById('snapshot');
    var player = document.getElementById('player');
    var button = document.getElementById('capture');

    // Handle successful webcam image sample
    var handleSuccess = function(stream) {
        player.srcObject = stream;
    }

    button.addEventListener('click', function() {
        var context = snapshot.getContext('2d');
        context.drawImage(player, 0, 0, snapshot.width, snapshot.height);
        $.ajax({
            type: "POST",
            url: "/upload",
            data: {
                imgBase64: snapshot.toDataURL()
            }
        }).done(function(o) {
            console.log("saved");
        });
    });

    navigator.mediaDevices.getUserMedia({video: true}).then(handleSuccess);
});

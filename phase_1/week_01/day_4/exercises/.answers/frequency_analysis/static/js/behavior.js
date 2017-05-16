$(document).ready(function() {
        console.log("teucer.js: The document is ready");

        $("seeker").on("submit", function(event) {
                event.preventDefault();
                data = $("seeker").serialize();
                $.ajax({
                        method: "POST",
                        url: "/seek",
                        data: data,
                        success: function(response) {
                                console.log(response);
                        }
                });
        });
});

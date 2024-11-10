let canvas = document.getElementById('drawingCanvas');
        let ctx = canvas.getContext('2d');
        let lastX = null;
        let lastY = null;
        ctx.lineWidth = 3;
        // Establish a single EventSource connection to receive coordinates
        const source = new EventSource('/coordinates');
        source.onmessage = function(event) {
            const data = event.data.split(',');
           
            const x = data[0];
            const y = data[1];
            // Draw on canvas if we have valid coordinates
            if(x === "w"){
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 30;
                return;
            }
            if(x === "l"){
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 3;
                return
            }
            if(x === "b" && y === "b"){
                ctx.clearRect(0,0,canvas.width,canvas.height);
                let lastX = null;
                let lastY = null;
                return;
            }
            if (x === "a")  {
            lastX = null;
            lastY = null;
            } else {
                const parsedX = parseFloat(x);
                const parsedY = parseFloat(y);
            // Draw on canvas if we have valid coordinates and valid previous coordinates
            if (lastX !== null && lastY !== null) {
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }

        }

            // Update the last coordinates
            lastX = x;
            lastY = y;

            // Display the current coordinates in the div
            document.getElementById('coordinates').innerText = `Index Finger: x: ${x}, y: ${y}`;
        };

        // Reset the canvas on click
        canvas.addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            lastX = null;
            lastY = null;
        });
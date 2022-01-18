session = "";
feeds = "";

$( document ).ready(function() {
  main();
});

async function main() {
  try {
    session = await ort.InferenceSession.create('./mosaic-8.onnx');
    $('#info').hide();
    $('#main').show();
    $('#run_btn').hide()
  } catch (e) {
    document.write(`failed to inference ONNX model: ${e}.`);
  }
}

let inputFile = async function(event) {
  $('#info').show();
  $('#info').text('Click Run to Continue, browser may freeze for a few seconds.')
  output_src = URL.createObjectURL(event.target.files[0]);
  var img = new Image();
  img.src = output_src
  img.onload = function() {
    $('#file_selector').hide()
    var canvas = document.getElementById('preview');
    ctx = canvas.getContext('2d');
    w = canvas.width
    h = canvas.height
    ctx.drawImage(img,0,0,img.width,img.height,0,0,w,h);
    $('#run_btn').show()
  }
}

let run = async function(event) {
  $('#run_btn').hide()
  var canvas = document.getElementById('preview');
  ctx = canvas.getContext('2d');
  const float32Data = new Float32Array(3*h*w);
  for (let i = 0; i < w; i+=1) {
    for (let j = 0; j < h; j+=1) {
      var pixel = ctx.getImageData(i, j, 1, 1).data;
      float32Data[0*h*w+j*w+i] = pixel[0]
      float32Data[1*h*w+j*w+i] = pixel[1]
      float32Data[2*h*w+j*w+i] = pixel[2]
    }
  }
  input_ts = new ort.Tensor('float32', float32Data, [1,3,h,w]);
  output_name = session.outputNames[0];
  input_name = session.inputNames[0];
  feeds = { 'input1': input_ts };
  const results = await session.run(feeds);
  const result_data = results.output1.data;
  result_h = results.output1.dims[2]
  result_w = results.output1.dims[3]
  result_img_data = new Uint8ClampedArray([result_h,result_w,4]);
  step0 = result_w*result_h;
  canvas = document.getElementById('result');
  ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (var i = 0; i < data.length; i += 4) {
    iw = (i/4)%result_w;
    ih = Math.floor(i/4/result_h)
    data[i] = result_data[0*step0 + ih*result_w + iw];
    data[i+1] = result_data[1*step0 + ih*result_w + iw];
    data[i+2] = result_data[2*step0 + ih*result_w + iw];
    data[i + 3] = 255; // alpha
  }
  ctx.putImageData(imageData, 0, 0);
  $('#info').text('Done. Refresh page to try again.');
};

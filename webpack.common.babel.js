import { join, resolve } from 'path';

const include = join(__dirname, 'src');

export default {
  entry: ['babel-polyfill', './src/index.js'],
  output: {
    path: resolve(__dirname, 'dist'),
    publicPath: '/',
    libraryTarget: 'umd',
    filename: 'pixelity.js',
    library: 'pixelity',
  },
  module: {
    rules: [
      {
        enforce: 'pre',
        test: /\.js$/,
        exclude: /node_modules/
      },
      {
        test: /\.js$/,
        loader: 'babel-loader',
        include,
      },
    ],
  }
};
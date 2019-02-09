import { join } from 'path';
import merge from 'webpack-merge';
import common from './webpack.common.babel';
import HtmlWebpackPlugin from 'html-webpack-plugin';

export default merge(common, {
  mode: 'development',
  devtool: 'inline-source-map',
  devServer: {
    watchContentBase: true,
    contentBase: join(__dirname, './build')
  },
  plugins: []
})
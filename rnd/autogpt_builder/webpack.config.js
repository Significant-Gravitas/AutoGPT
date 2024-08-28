module.exports = {
  devServer: {
    proxy: {
      "/graphs": "http://localhost:8000",
    },
  },
};

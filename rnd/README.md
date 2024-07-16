This is a gude to setting up and running the AutoGPT Server and Builder. This tutorial will cover downloading the necessary files, setting up the server, and testing the system.

https://github.com/user-attachments/assets/fd0d0f35-3155-4263-b575-ba3efb126cb4

1. Navigate to the AutoGPT GitHub repository.
2. Click the "Code" button, then select "Download ZIP".
3. Once downloaded, extract the ZIP file to a folder of your choice.

4. Open the extracted folder and navigate to the "rnd" directory.
5. Enter the "AutoGPT server" folder.
6. Open a terminal window in this directory.
7. Locate and open the README file in the AutoGPT server folder.
8. Copy and paste each command from the README into your terminal.
   - Important: Wait for each command to finish before running the next one.
9. If all commands run without errors, enter the final command: `poetry run app`
10. You should now see the server running in your terminal.

11. Navigate back to the "rnd" folder.
12. Open the "AutoGPT builder" folder.
13. Open the README file in this folder.
14. In your terminal, run the following commands:
    ```
    npm install
    ```
    ```
    npm run dev
    ```

15. Once the front-end is running, click the link to navigate to `localhost:3000`.
16. Click on the "Build" option.
17. Add a few blocks to test the functionality.
18. Connect the blocks together.
19. Click "Run".
20. Check your terminal window - you should see that the server has received the request, is processing it, and has executed it.

And there you have it! You've successfully set up and tested AutoGPT.

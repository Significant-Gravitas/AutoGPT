import { expect, test } from "./fixtures";
import {
  allTab,
  backToLibrary,
  breadcrumb,
  clearSearch,
  createFolder,
  createFolderInDialog,
  createWorkflows,
  deleteFolderInDialog,
  editFolderInDialog,
  expectEditDialogShows,
  expectListed,
  favoritesTab,
  FOLDER_COLOR,
  folderCard,
  listedWorkflows,
  moveInDialog,
  moveWorkflows,
  observe,
  openFolder,
  openLibrary,
  orderBy,
  readEntry,
  readFavoritesCount,
  readFolder,
  readFolders,
  removeFromFolder,
  runToEnd,
  search,
  setFavorite,
  signOutAndBackIn,
  SORTS,
  sortBy,
  sortMenu,
  uniqueStamp,
  workflowCard,
} from "./library-organization";

test.describe("Library organization journey", () => {
  test("a folder holds the workflows moved into it, keeps its new name and colour through a reload and a new session, and hands them back when it is deleted", async ({
    page,
    journeyUser,
  }, testInfo) => {
    const stamp = uniqueStamp();
    const workflows = await createWorkflows(page, stamp, [
      "Alpha",
      "Beta",
      "Gamma",
    ]);
    const [alpha, beta, gamma] = workflows;
    await openLibrary(page);
    await expectListed(
      page,
      workflows,
      workflows,
      "the Library should list the new workflows",
    );

    const folder = await createFolderInDialog(page, {
      name: `Folder ${stamp}`,
      color: "Blue",
    });
    const card = folderCard(page, folder.id);
    await expect(card.getByTestId("library-folder-name")).toHaveText(
      folder.name,
    );
    await expect(card.getByTestId("library-folder-agent-count")).toHaveText(
      "0 workflows",
    );

    const allTabBefore = await allTab(page).innerText();
    await moveInDialog(page, alpha, folder);
    await moveInDialog(page, beta, folder);
    await expect(
      card.getByTestId("library-folder-agent-count"),
      "the folder should count the two workflows moved into it",
    ).toHaveText("2 workflows");
    await expectListed(
      page,
      workflows,
      [gamma],
      "the main list should keep only the workflow left out of the folder",
    );
    expect(
      (await readFolders(page)).find((listed) => listed.id === folder.id)
        ?.agent_count,
    ).toBe(2);
    // Recorded, not asserted: give the count a moment to follow the moves.
    await expect
      .poll(() => allTab(page).innerText(), { timeout: 5_000 })
      .not.toBe(allTabBefore)
      .catch(() => {});
    observe(
      testInfo,
      `with 2 of 3 workflows in a folder, the All tab reads ` +
        `"${await allTab(page).innerText()}"`,
    );

    await openFolder(page, folder);
    await expect(breadcrumb(page)).toContainText(folder.name);
    await expectListed(
      page,
      workflows,
      [alpha, beta],
      "the folder should show exactly the two workflows moved into it",
    );
    await expect(page.getByTestId("library-agent-card")).toHaveCount(2);
    await backToLibrary(page);

    const renamed = `Renamed ${stamp}`;
    const edited = await editFolderInDialog(page, folder, {
      name: renamed,
      color: "Purple",
    });
    expect(edited).toMatchObject({
      name: renamed,
      color: FOLDER_COLOR.Purple,
    });
    expect(edited.icon).not.toBe(folder.icon);
    await expect(card.getByTestId("library-folder-name")).toHaveText(renamed);

    await page.reload();
    await expect(card.getByTestId("library-folder-name")).toHaveText(renamed);
    await expect(card.getByTestId("library-folder-agent-count")).toHaveText(
      "2 workflows",
    );
    await expectEditDialogShows(page, edited, "Purple");

    await signOutAndBackIn(page, journeyUser);
    await openLibrary(page);
    await expect(card.getByTestId("library-folder-name")).toHaveText(renamed);
    await expect(card.getByTestId("library-folder-agent-count")).toHaveText(
      "2 workflows",
    );
    expect(
      (await readFolder(page, folder.id)).folder,
      "the backend should keep every edit",
    ).toMatchObject({
      name: renamed,
      color: FOLDER_COLOR.Purple,
      icon: edited.icon,
    });
    await openFolder(page, edited);
    await expectListed(
      page,
      workflows,
      [alpha, beta],
      "the folder should still hold its two workflows in a new session",
    );
    await backToLibrary(page);

    await deleteFolderInDialog(page, edited);
    await expectListed(
      page,
      workflows,
      workflows,
      "deleting the folder should hand its workflows back to the main list",
    );
    for (const workflow of workflows) {
      expect(
        (await readEntry(page, workflow)).folder_id,
        `${workflow.name} should be out of the deleted folder`,
      ).toBeNull();
    }
    expect((await readFolder(page, folder.id)).status).toBe(404);
    await page.reload();
    await expectListed(
      page,
      workflows,
      workflows,
      "after a reload, the workflows should still be in the main list",
    );
    await expect(folderCard(page, folder.id)).toHaveCount(0);
  });

  test("a favorite inside a folder shows on the Favorites tab until unfavorited, search finds workflows in folders, and each sort orders them by its own time", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const stamp = uniqueStamp();
    const workflows = await createWorkflows(page, stamp, [
      "Alpha",
      "Beta",
      "Gamma",
      "Delta",
    ]);
    const [alpha, beta, gamma, delta] = workflows;
    const folder = await createFolder(page, {
      name: `Folder ${stamp}`,
      color: "Green",
      icon: "📁",
    });
    await moveWorkflows(page, [beta, gamma], folder.id);
    // Run times for the Last Run sort: Alpha ran last, then Gamma; Beta and
    // Delta never ran.
    await runToEnd(page, gamma);
    await runToEnd(page, alpha);
    const favoritesBefore = await readFavoritesCount(page);

    await openLibrary(page);
    await expectListed(
      page,
      workflows,
      [alpha, delta],
      "the main list should leave out the workflows in the folder",
    );
    await openFolder(page, folder);
    await expectListed(
      page,
      workflows,
      [beta, gamma],
      "the folder's workflows",
    );
    await setFavorite(page, beta, true);
    await expect(
      workflowCard(page, beta).getByRole("button", {
        name: "Remove from favorites",
      }),
    ).toBeVisible();
    await backToLibrary(page);

    await expect(
      favoritesTab(page),
      "the Favorites tab should count the new favorite",
    ).toHaveText(`Favorites ${favoritesBefore + 1}`);
    await favoritesTab(page).click();
    await expectListed(
      page,
      workflows,
      [beta],
      "the Favorites tab should list the favorite from inside the folder",
    );
    await expect(folderCard(page, folder.id)).toHaveCount(0);
    await setFavorite(page, beta, false);
    await expectListed(
      page,
      workflows,
      [],
      "an unfavorited workflow should leave the Favorites tab",
    );
    expect(await readFavoritesCount(page)).toBe(favoritesBefore);
    if (favoritesBefore === 0) {
      await expect(page.getByText("No favorite workflows yet")).toBeVisible();
      await expect(favoritesTab(page)).toHaveText("Favorites");
    }
    expect(
      await readEntry(page, beta),
      "favoriting should leave the workflow in its folder",
    ).toMatchObject({ is_favorite: false, folder_id: folder.id });
    await allTab(page).click();
    await expect(allTab(page)).toHaveAttribute("aria-selected", "true");

    await search(page, beta.name);
    await expectListed(
      page,
      workflows,
      [beta],
      "searching the main list should find a workflow inside a folder",
    );
    await expect(page.getByTestId("library-agent-card")).toHaveCount(1);
    await expect(folderCard(page, folder.id)).toHaveCount(0);
    await clearSearch(page, folderCard(page, folder.id));
    await openFolder(page, folder);
    await search(page, delta.name);
    await expect(
      page.locator('[data-slot="loading-spinner"]'),
      "the search inside the folder should finish loading",
    ).toHaveCount(0);
    // Neither of the folder's workflows matches, whatever the search spans.
    await expectListed(page, [beta, gamma], [], "the search inside the folder");
    observe(
      testInfo,
      `inside the folder, a search for a workflow outside it lists: ` +
        `${JSON.stringify(await listedWorkflows(page, workflows))}`,
    );
    await clearSearch(page, workflowCard(page, beta));
    await backToLibrary(page);

    // Every workflow of this test, wherever it is filed, in each sort.
    await search(page, stamp);
    await expectListed(page, workflows, workflows, "the search for all four");
    const orders: string[][] = [];
    for (const label of Object.keys(SORTS) as (keyof typeof SORTS)[]) {
      await sortBy(page, label);
      const expected = await orderBy(page, workflows, label);
      orders.push(expected);
      await expect
        .poll(() => listedWorkflows(page, workflows), {
          message: `${label} should order the workflows by its own time`,
        })
        .toEqual(expected);
      await expect(sortMenu(page)).toHaveText(label);
    }
    expect(
      new Set(orders.map((order) => order.join())).size,
      "the journey's workflows should come out in three different orders",
    ).toBe(3);
    expect(orders[2].slice(0, 2), "the two runs, newest first").toEqual([
      alpha.name,
      gamma.name,
    ]);

    // The sort is in the URL, so a reload keeps the order. Text typed before
    // the page hydrates never reaches the search, so wait for a card first.
    await page.reload();
    await expect(workflowCard(page, alpha)).toBeVisible();
    await search(page, stamp);
    await expect
      .poll(() => listedWorkflows(page, workflows), {
        message: "a reload should keep the Last Run order",
      })
      .toEqual(orders[2]);
  });

  test("a subfolder opens from its parent, a workflow moves back out of a folder, and deleting a parent hands back everything under it", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const stamp = uniqueStamp();
    const workflows = await createWorkflows(page, stamp, ["Alpha", "Beta"]);
    const [alpha, beta] = workflows;
    const parent = await createFolder(page, {
      name: `Parent ${stamp}`,
      color: "Orange",
      icon: "📁",
    });
    const child = await createFolder(page, {
      name: `Child ${stamp}`,
      color: "Blue",
      icon: "📂",
      parentID: parent.id,
    });
    await moveWorkflows(page, [alpha], parent.id);
    await moveWorkflows(page, [beta], child.id);

    await openLibrary(page);
    await expect(folderCard(page, parent.id)).toBeVisible();
    await expect(
      folderCard(page, child.id),
      "a subfolder should show only inside its parent",
    ).toHaveCount(0);
    await expectListed(page, workflows, [], "both workflows are filed away");
    observe(
      testInfo,
      `a parent holding 1 workflow and a subfolder with 1 more reads ` +
        `"${await folderCard(page, parent.id).getByTestId("library-folder-agent-count").innerText()}"`,
    );

    await openFolder(page, parent);
    await expect(folderCard(page, child.id)).toBeVisible();
    await expect(
      folderCard(page, child.id).getByTestId("library-folder-agent-count"),
    ).toHaveText("1 workflow");
    await expectListed(page, workflows, [alpha], "the parent's own workflow");

    await openFolder(page, child);
    await expect(breadcrumb(page)).toContainText(child.name);
    observe(
      testInfo,
      `a subfolder's breadcrumb reads "${await breadcrumb(page).innerText()}"`,
    );
    await expectListed(page, workflows, [beta], "the subfolder's workflow");
    await removeFromFolder(page, beta);
    await expectListed(
      page,
      workflows,
      [],
      "a workflow removed from a folder should leave it",
    );
    await expect(page.getByTestId("library-agent-card")).toHaveCount(0);
    const belowBreadcrumb = await breadcrumb(page)
      .locator("xpath=following-sibling::*")
      .allInnerTexts();
    observe(
      testInfo,
      `an emptied folder shows below its breadcrumb: ` +
        `${JSON.stringify(belowBreadcrumb.join(" ").replace(/\s+/g, " ").trim())}`,
    );
    await backToLibrary(page);
    await expectListed(
      page,
      workflows,
      [beta],
      "the removed workflow should be back in the main list",
    );
    expect((await readEntry(page, beta)).folder_id).toBeNull();

    await moveWorkflows(page, [beta], child.id);
    await openLibrary(page);
    await expectListed(
      page,
      workflows,
      [],
      "a workflow filed in the subfolder again should leave the main list",
    );

    await deleteFolderInDialog(page, parent);
    await expectListed(
      page,
      workflows,
      workflows,
      "deleting the parent should hand back the workflows under it, its subfolder's included",
    );
    expect((await readEntry(page, alpha)).folder_id).toBeNull();
    expect((await readEntry(page, beta)).folder_id).toBeNull();
    expect(
      (await readFolder(page, child.id)).status,
      "the subfolder should go with its parent",
    ).toBe(404);
  });
});

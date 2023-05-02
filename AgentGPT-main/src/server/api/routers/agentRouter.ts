import { z } from "zod";

import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { prisma } from "../../db";
import { messageParser } from "../../../types/agentTypes";

const saveAgentParser = z.object({
  name: z.string(),
  goal: z.string(),
  tasks: z.array(messageParser),
});

export const agentRouter = createTRPCRouter({
  create: protectedProcedure
    .input(saveAgentParser)
    .mutation(async ({ input, ctx }) => {
      const agent = await prisma.agent.create({
        data: {
          name: input.name,
          goal: input.goal,
          userId: ctx.session?.user?.id,
        },
      });

      const all = input.tasks.map((e, i) => {
        return prisma.agentTask.create({
          data: {
            agentId: agent.id,
            type: e.type,
            info: e.info,
            value: e.value,
            sort: i,
          },
        });
      });

      await Promise.all(all);
      return agent;
    }),
  getAll: protectedProcedure.query(async ({ ctx }) => {
    return prisma.agent.findMany({
      where: {
        userId: ctx.session?.user?.id,
        deleteDate: null,
      },
      orderBy: { createDate: "desc" },
      take: 20,
    });
  }),
  findById: publicProcedure.input(z.string()).query(async ({ input, ctx }) => {
    const agent = await prisma.agent.findFirstOrThrow({
      where: { id: input, deleteDate: null },
      include: {
        tasks: {
          orderBy: {
            sort: "asc",
          },
        },
      },
    });

    return {
      ...agent,
    };
  }),
  deleteById: protectedProcedure
    .input(z.string())
    .mutation(async ({ input, ctx }) => {
      await prisma.agent.updateMany({
        where: { id: input, userId: ctx.session?.user?.id },
        data: {
          deleteDate: new Date(),
        },
      });
    }),
});

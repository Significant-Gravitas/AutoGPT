"use client";
import React, { useState } from 'react';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const AgentFlowList = ({ flows, onSelectFlow }) => (
  <Card>
    <CardHeader>
      <CardTitle>Agent Flows</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Last Run</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {flows.map((flow) => (
            <TableRow key={flow.id} onClick={() => onSelectFlow(flow)} className="cursor-pointer">
              <TableCell>{flow.name}</TableCell>
              <TableCell>{flow.status}</TableCell>
              <TableCell>{flow.lastRun}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

const FlowRunsList = ({ runs }) => (
  <Card>
    <CardHeader>
      <CardTitle>Flow Runs</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell>{run.time}</TableCell>
              <TableCell>{run.status}</TableCell>
              <TableCell>{run.duration}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

const FlowStats = ({ stats }) => (
  <Card>
    <CardHeader>
      <CardTitle>Flow Statistics</CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={stats}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

const Monitor = () => {
  const [selectedFlow, setSelectedFlow] = useState(null);
  
  // Mock data
  const flows = [
    { id: 1, name: 'JARVIS', status: 'Waiting for input', lastRun: '5 minutes ago' },
    { id: 2, name: 'Time machine', status: 'Crashed', lastRun: '10 minutes ago' },
    { id: 3, name: 'BlueSky digest', status: 'Running', lastRun: '2 minutes ago' },
  ];

  const runs = [
    { id: 1, time: '12:34', status: 'Success', duration: '1m 26s' },
    { id: 2, time: '11:49', status: 'Success', duration: '55s' },
    { id: 3, time: '11:23', status: 'Success', duration: '48s' },
  ];

  const stats = [
    { name: 'Last 24 Hours', value: 16 },
    { name: 'Last 30 Days', value: 106 },
  ];

  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <AgentFlowList flows={flows} onSelectFlow={setSelectedFlow} />
      </div>
      <div className="space-y-4">
        <FlowRunsList runs={runs} />
        <FlowStats stats={stats} />
      </div>
      {selectedFlow && (
        <div className="col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>{selectedFlow.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <Button>Edit Flow</Button>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default Monitor;
